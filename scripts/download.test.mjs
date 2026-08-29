import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { createServer } from "node:http";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { benchmarkRoutes, createSpeedometer, downloadFile, downloadInternals } from "./download.mjs";

test("content ranges expose total file size", () => {
  assert.deepEqual(downloadInternals.contentRange("bytes 0-65535/123456"), { start: 0, end: 65535, total: 123456 });
  assert.equal(downloadInternals.contentRange(null), null);
});

test("large downloads use the requested bounded part count", () => {
  assert.equal(downloadInternals.partCount(8 * 1024 ** 2, 8, 64 * 1024 ** 2), 1);
  assert.equal(downloadInternals.partCount(512 * 1024 ** 2, 8, 64 * 1024 ** 2), 8);
});

test("route benchmarks select the responsive target item", async () => {
  const fetcher = async (url) => {
    if (url.endsWith("slow")) await new Promise((resolve) => setTimeout(resolve, 30));
    return new Response(Buffer.alloc(1024), {
      status: 206,
      headers: { "Content-Length": "1024", "Content-Range": "bytes 0-1023/4096" },
    });
  };
  const ranked = await benchmarkRoutes([
    { id: "slow", label: "slow", url: "https://example.test/slow" },
    { id: "fast", label: "fast", url: "https://example.test/fast" },
  ], { fetcher, sampleBytes: 1024 });
  assert.equal(ranked[0].id, "fast");
  assert.equal(ranked[0].supportsRanges, true);
});

test("range downloads land directly in the output file without a merge pass", async () => {
  const payload = randomBytes(2 * 1024 ** 2);
  let rangeRequests = 0;
  const server = createServer((request, response) => {
    const match = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    if (!match) {
      response.writeHead(200, { "Content-Length": payload.length });
      response.end(payload);
      return;
    }
    rangeRequests += 1;
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), payload.length - 1);
    response.writeHead(206, {
      "Accept-Ranges": "bytes",
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${payload.length}`,
    });
    response.end(payload.subarray(start, end + 1));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-download-"));
  const destination = path.join(directory, "fixture.bin");
  try {
    const digest = createHash("sha256").update(payload).digest("hex");
    const result = await downloadFile({
      routes: [{ id: "local", label: "local", url: `http://127.0.0.1:${server.address().port}/fixture.bin` }],
      destination,
      expectedSha256: digest,
      connections: 4,
      thresholdBytes: 1,
    });
    assert.equal(result.connections, 4);
    assert.ok(rangeRequests >= 5);
    assert.deepEqual(await readFile(destination), payload);
    // Every segment writes at its absolute offset in one file. The per-connection `.part.N` files
    // the previous engine merged at the end cost a second full copy of the wheel on disk and a
    // second full pass over it — a real minute on a 2.6 GB PyTorch download.
    assert.deepEqual(await readdir(directory), ["fixture.bin"]);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});

test("interrupted range segments resume from their saved offsets", async () => {
  const payload = randomBytes(512 * 1024);
  const segmentSize = payload.length / 4;
  let allowSecondSegment = false;
  let resumedOffset = null;
  // The link writes a little once and then goes dead. Only the dead part exhausts the retry
  // budget — a connection that keeps delivering bytes is not a failed one, however often it has
  // to be re-established.
  let truncatedAttempts = 0;
  const server = createServer((request, response) => {
    const match = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    if (!match) {
      response.writeHead(200, { "Content-Length": payload.length });
      response.end(payload);
      return;
    }
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), payload.length - 1);
    response.writeHead(206, {
      "Accept-Ranges": "bytes",
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${payload.length}`,
    });
    if (start >= segmentSize && start < segmentSize * 2 && !allowSecondSegment) {
      truncatedAttempts += 1;
      if (truncatedAttempts === 1) response.write(payload.subarray(start, Math.min(end + 1, start + 4096)));
      setTimeout(() => response.destroy(), 20);
      return;
    }
    if (allowSecondSegment && start > segmentSize && start < segmentSize * 2) resumedOffset = start;
    response.end(payload.subarray(start, end + 1));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-resume-download-"));
  const destination = path.join(directory, "fixture.bin");
  const options = {
    routes: [{ id: "local", label: "local", url: `http://127.0.0.1:${server.address().port}/fixture.bin` }],
    destination,
    expectedSha256: createHash("sha256").update(payload).digest("hex"),
    connections: 4,
    thresholdBytes: 1,
  };
  try {
    await assert.rejects(downloadFile(options));
    // Four dead attempts is what ends it, and the one that delivered bytes is not among them.
    assert.ok(truncatedAttempts >= 5, `expected the dead link to be retried, got ${truncatedAttempts}`);
    allowSecondSegment = true;
    await downloadFile(options);
    assert.ok(resumedOffset > segmentSize);
    assert.deepEqual(await readFile(destination), payload);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});

test("a mirror that stalls but keeps delivering bytes is not abandoned", async () => {
  // Model mirrors go quiet under load and drop the connection, then serve the next range fine.
  // Counting those against a fixed retry budget threw away multi-gigabyte transfers that were
  // still advancing — the .part files were intact and the next attempt would have finished them.
  const payload = randomBytes(256 * 1024);
  let interruptions = 0;
  // The first ranged request is the route benchmark, which has to answer normally or the route is
  // ranked unusable before a byte of the file is fetched.
  let served = 0;
  const server = createServer((request, response) => {
    const match = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    if (!match) {
      response.writeHead(200, { "Content-Length": payload.length });
      response.end(payload);
      return;
    }
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), payload.length - 1);
    response.writeHead(206, {
      "Accept-Ranges": "bytes",
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${payload.length}`,
    });
    // Every request delivers a slice and dies: eight round trips per segment, well past the four
    // a fixed budget allowed.
    served += 1;
    if (served > 1 && end - start + 1 > 8192) {
      interruptions += 1;
      response.write(payload.subarray(start, start + 8192));
      setTimeout(() => response.destroy(), 5);
      return;
    }
    response.end(payload.subarray(start, end + 1));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-stall-download-"));
  const destination = path.join(directory, "fixture.bin");
  try {
    await downloadFile({
      routes: [{ id: "local", label: "local", url: `http://127.0.0.1:${server.address().port}/fixture.bin` }],
      destination,
      expectedSha256: createHash("sha256").update(payload).digest("hex"),
      connections: 4,
      thresholdBytes: 1,
    });
    assert.ok(interruptions > 4, `expected more interruptions than the old budget, got ${interruptions}`);
    assert.deepEqual(await readFile(destination), payload);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});

test("one file is fetched from every mirror that agrees about it, not only from the fastest", async () => {
  // This is the complaint the engine exists to answer: ranking once and committing the whole
  // transfer to the winner means a mirror that measured well for 2 MB decides the next twenty
  // minutes. Segments are dealt across the mirrors instead, so each one carries what it can.
  const payload = randomBytes(1024 ** 2);
  const served = new Map();
  const makeServer = (id, delayMs) => createServer((request, response) => {
    const match = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    const start = match ? Number(match[1]) : 0;
    const end = match ? Math.min(Number(match[2]), payload.length - 1) : payload.length - 1;
    const body = payload.subarray(start, end + 1);
    const finish = () => {
      served.set(id, (served.get(id) || 0) + body.length);
      response.writeHead(match ? 206 : 200, {
        "Accept-Ranges": "bytes",
        "Content-Length": body.length,
        ...(match ? { "Content-Range": `bytes ${start}-${end}/${payload.length}` } : {}),
      });
      response.end(body);
    };
    if (delayMs) setTimeout(finish, delayMs); else finish();
  });

  const servers = [makeServer("fast", 0), makeServer("slow", 15)];
  await Promise.all(servers.map((server) => new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))));
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-multi-source-"));
  const destination = path.join(directory, "fixture.bin");
  try {
    const routeSummary = [];
    const result = await downloadFile({
      routes: [
        { id: "fast", label: "fast", url: `http://127.0.0.1:${servers[0].address().port}/fixture.bin` },
        { id: "slow", label: "slow", url: `http://127.0.0.1:${servers[1].address().port}/fixture.bin` },
      ],
      destination,
      expectedSha256: createHash("sha256").update(payload).digest("hex"),
      connections: 4,
      thresholdBytes: 1,
      onRoute: (route) => routeSummary.push(route),
    });
    assert.deepEqual(await readFile(destination), payload);
    // Both mirrors carried part of the file, and the panel was told so rather than being shown one
    // label while two connections ran.
    assert.ok(served.get("fast") > 0, "the fast mirror served nothing");
    assert.ok(served.get("slow") > 0, "the slow mirror served nothing");
    assert.equal(routeSummary[0].sources.length, 2);
    assert.equal(result.connections, 4);
  } finally {
    await Promise.all(servers.map((server) => new Promise((resolve) => server.close(resolve))));
    await rm(directory, { recursive: true, force: true });
  }
});

test("a mirror serving a different length is kept out of the shared file", () => {
  // Same filename, different build. Splitting one download across both interleaves two files, and
  // without a published digest the result is a corrupt archive that looks complete.
  const routes = [
    { id: "a", label: "a", url: "https://a.test/w.whl", ok: true, supportsRanges: true, totalBytes: 1000 },
    { id: "b", label: "b", url: "https://b.test/w.whl", ok: true, supportsRanges: true, totalBytes: 1000 },
    { id: "c", label: "c", url: "https://c.test/w.whl", ok: true, supportsRanges: true, totalBytes: 1001 },
    { id: "d", label: "d", url: "https://d.test/w.whl", ok: true, supportsRanges: false, totalBytes: 1000 },
  ];
  assert.deepEqual(downloadInternals.consistentRangedRoutes(routes).map((route) => route.id), ["a", "b"]);
  // A lone route is never rejected for disagreeing with itself.
  assert.deepEqual(downloadInternals.consistentRangedRoutes([routes[2]]).map((route) => route.id), ["c"]);
});

test("routes are ranked by how long the file will take, not by who answers first", () => {
  const nearby = { ok: true, latencyMs: 5, speedBps: 2 * 1024 ** 2, supportsRanges: true };
  const distant = { ok: true, latencyMs: 180, speedBps: 40 * 1024 ** 2, supportsRanges: true };
  // For a 96 KB index page the low-latency mirror is genuinely the right answer...
  assert.ok(downloadInternals.predictedSeconds(nearby, 96 * 1024) < downloadInternals.predictedSeconds(distant, 96 * 1024));
  // ...and for a 2.6 GB wheel it is the wrong one by twenty minutes. One round trip is noise at
  // that size; bandwidth is the whole answer.
  assert.ok(downloadInternals.predictedSeconds(distant, 2.6 * 1024 ** 3) < downloadInternals.predictedSeconds(nearby, 2.6 * 1024 ** 3));
  // A route without range support cannot be segmented or resumed, so an equal measurement is not
  // an equal prospect.
  assert.ok(
    downloadInternals.predictedSeconds({ ...nearby, supportsRanges: false }, 1024 ** 2)
    > downloadInternals.predictedSeconds(nearby, 1024 ** 2),
  );
  assert.equal(downloadInternals.predictedSeconds({ ok: false }, 1024), Number.POSITIVE_INFINITY);
});

test("the benchmark times the link rather than the handshake", async () => {
  // TLS and TCP slow start land in the opening tens of kilobytes. Timing them measures the
  // connection setup, which is how a nearby mirror on a thin pipe used to beat a fast distant one.
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const fetcher = async () => new Response(ReadableStream.from((async function* body() {
    await sleep(200);
    yield new Uint8Array(32 * 1024);   // the handshake burst
    await sleep(20);
    yield new Uint8Array(480 * 1024);  // then line rate
  })()), {
    status: 206,
    headers: { "Content-Range": `bytes 0-${512 * 1024 - 1}/${8 * 1024 ** 2}` },
  });
  const startedAt = performance.now();
  const [route] = await benchmarkRoutes([
    { id: "local", label: "local", url: "https://local.test/w.whl" },
  ], { fetcher, warmupBytes: 32 * 1024, sampleBytes: 512 * 1024, sampleMs: 5000 });
  const wholeSampleRate = 512 * 1024 / Math.max((performance.now() - startedAt) / 1000, 0.001);
  assert.equal(route.ok, true);
  assert.equal(route.supportsRanges, true);
  assert.equal(route.totalBytes, 8 * 1024 ** 2);
  // The absolute number depends on the host's timer granularity; the claim under test does not.
  // Whole-sample timing charges the 200 ms handshake to the link and reports several times less
  // than the 480 KB that actually crossed it.
  assert.ok(
    route.speedBps > wholeSampleRate * 2,
    `steady-state ${Math.round(route.speedBps)} B/s did not beat whole-sample ${Math.round(wholeSampleRate)} B/s`,
  );
});

test("a body too small or too fast to separate falls back to timing the whole sample", async () => {
  // One chunk carrying both the warm-up and the rest leaves no steady window to divide by, and an
  // arbitrarily large reported rate would then decide where gigabytes come from.
  const fetcher = async () => new Response(Buffer.alloc(512 * 1024), {
    status: 206,
    headers: { "Content-Range": `bytes 0-${512 * 1024 - 1}/${512 * 1024}` },
  });
  const [route] = await benchmarkRoutes([{ id: "a", label: "a", url: "https://a.test/w" }], {
    fetcher, warmupBytes: 32 * 1024, sampleBytes: 512 * 1024,
  });
  assert.equal(route.ok, true);
  assert.ok(Number.isFinite(route.speedBps) && route.speedBps > 0);
});

test("a probe is scaled to the file it is choosing a mirror for", async () => {
  // Sampling a fixed 2 MB to pick a mirror for a 6 MB YOLO model would spend more on the probe than
  // on the download.
  let requested = 0;
  const fetcher = async () => new Response(Buffer.alloc(6 * 1024 ** 2), {
    status: 206,
    headers: { "Content-Range": `bytes 0-${6 * 1024 ** 2 - 1}/${6 * 1024 ** 2}` },
  });
  const [route] = await benchmarkRoutes([{ id: "a", label: "a", url: "https://a.test/model.pt" }], {
    fetcher: async (...args) => {
      requested += 1;
      return fetcher(...args);
    },
    sampleBytes: 2 * 1024 ** 2,
  });
  assert.equal(requested, 1);
  // One 6 MB body arrives in a single chunk here, so the cap is asserted through the plan rather
  // than the read: a twentieth of 6 MB is under the warm-up floor, so the floor is what applies.
  assert.ok(route.sampledBytes > 0);
  assert.equal(route.totalBytes, 6 * 1024 ** 2);
});

test("segments are split so an idle connection can take over a laggard's tail", () => {
  // Equal ranges finish at unequal times once they are spread over mirrors of different speeds, and
  // the file is not done until the slowest one is.
  const segment = { start: 0, end: 40 * 1024 ** 2 - 1, downloaded: 8 * 1024 ** 2, switches: 0 };
  const stolen = downloadInternals.splitSegment(segment);
  assert.equal(segment.end, 24 * 1024 ** 2 - 1);
  assert.equal(stolen.start, 24 * 1024 ** 2);
  assert.equal(stolen.end, 40 * 1024 ** 2 - 1);
  assert.equal(stolen.downloaded, 0);
  // Nothing is lost or duplicated across the split.
  assert.equal(segment.end - segment.start + 1 + (stolen.end - stolen.start + 1), 40 * 1024 ** 2);

  // A tail too small to be worth another handshake is left alone.
  assert.equal(downloadInternals.splitSegment({ start: 0, end: 1024 ** 2, downloaded: 0, switches: 0 }), null);
});

test("segments cover the file exactly, whatever the connection count", () => {
  for (const [total, connections] of [[1000, 4], [1001, 4], [7, 8], [64 * 1024 ** 2, 8]]) {
    const segments = downloadInternals.planSegments(total, connections);
    assert.equal(segments[0].start, 0);
    assert.equal(segments[segments.length - 1].end, total - 1);
    for (let index = 1; index < segments.length; index += 1) {
      assert.equal(segments[index].start, segments[index - 1].end + 1, `gap before segment ${index}`);
    }
    assert.ok(segments.length <= connections);
  }
});

test("a segment leaves a mirror that turns out slow in flight, while the move can still pay for itself", () => {
  // The benchmark is a 2 MB sample taken before the transfer. What it cannot see is a mirror that
  // answers well and then collapses under load — which is exactly the case that used to hold a
  // multi-gigabyte download hostage, because only an outright failure moved it on.
  let clock = 0;
  const a = { id: "a", label: "a", url: "https://a.test/w", speedBps: 20 * 1024 ** 2 };
  const b = { id: "b", label: "b", url: "https://b.test/w", speedBps: 20 * 1024 ** 2 };
  const pool = downloadInternals.createRoutePool([a, b], 2, { now: () => clock });
  pool.acquire(a);
  pool.acquire(b);
  pool.record(a, 0);
  pool.record(b, 0);
  clock = 10_000;
  pool.record(a, 400 * 1024 ** 2);  // 40 MB/s measured
  pool.record(b, 4 * 1024 ** 2);    // 0.4 MB/s measured, despite an identical benchmark

  const segment = { start: 0, end: 512 * 1024 ** 2 - 1, downloaded: 0, switches: 0 };
  assert.equal(pool.shouldLeave(b, segment), true, "a hundredfold faster alternative is worth moving to");
  assert.equal(segment.switches, 1);
  assert.equal(pool.shouldLeave(a, { ...segment, switches: 0 }), false, "the fastest mirror is never abandoned");

  // A tail that is nearly finished is not worth another handshake, however slow the link is.
  assert.equal(pool.shouldLeave(b, { start: 0, end: 1024 ** 2, downloaded: 0, switches: 0 }), false);

  // Two mirrors that measure alternately faster must not trade a segment back and forth instead of
  // downloading it.
  assert.equal(pool.shouldLeave(b, { ...segment, switches: 3 }), false);

  // With nowhere else to go, staying is the only option — killing the only live connection would
  // turn a slow download into no download.
  const alone = downloadInternals.createRoutePool([b], 2, { now: () => clock });
  assert.equal(alone.shouldLeave(b, { ...segment, switches: 0 }), false);
});

test("a mirror an order of magnitude slower is not given an equal share of the segments", () => {
  const fast = { id: "fast", label: "fast", url: "https://fast.test/w", speedBps: 50 * 1024 ** 2 };
  const usable = { id: "usable", label: "usable", url: "https://usable.test/w", speedBps: 20 * 1024 ** 2 };
  const crawling = { id: "crawl", label: "crawl", url: "https://crawl.test/w", speedBps: 64 * 1024 };
  const pool = downloadInternals.createRoutePool([fast, usable, crawling], 8);
  assert.deepEqual(pool.routes.map((route) => route.id), ["fast", "usable"]);
  // Handing a quarter of the file to a mirror 800 times slower would make it the tail that decides
  // when the download finishes; it stays available as a fallback rather than as a co-worker.
  assert.equal(pool.size, 2);
});

test("reported speed is what is arriving now, not what a resumed file already held", () => {
  // A transfer that resumes with 20 GB on disk used to divide that by a fraction of a second and
  // print an impossible rate, and a mirror that died ten minutes ago went on looking fast.
  const meter = createSpeedometer(1000);
  meter.record(20 * 1024 ** 3, 0);
  meter.record(20 * 1024 ** 3 + 1024 ** 2, 1000);
  assert.equal(meter.speed(1000), 1024 ** 2);
  // Nothing has arrived for another ten seconds: the reading falls rather than freezing.
  assert.ok(meter.speed(11000) < 1024 ** 2 / 8);
  assert.equal(createSpeedometer().speed(), 0);
});
