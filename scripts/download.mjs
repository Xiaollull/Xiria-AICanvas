import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const userAgent = "XirAI-Setup/0.1";

function contentRange(value) {
  const match = value?.match(/^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i);
  return match ? {
    start: Number(match[1]),
    end: Number(match[2]),
    total: match[3] === "*" ? 0 : Number(match[3]),
  } : null;
}

function partCount(totalBytes, requested, thresholdBytes) {
  if (!totalBytes || totalBytes < thresholdBytes || requested < 2) return 1;
  return Math.max(1, Math.min(requested, totalBytes));
}

// A throughput reading taken over a trailing window rather than over the whole transfer. The
// lifetime average is the wrong number to show: a download that resumes with 20 GB already on disk
// reports that 20 GB against a fraction of a second and prints an impossible speed, and a route
// that collapsed ten minutes ago still looks fast. The window forgets, which is what makes the
// reading usable both for the progress panel and for deciding a mirror is too slow to keep.
export function createSpeedometer(windowMs = 6000) {
  const samples = [];
  return {
    record(cumulativeBytes, now = performance.now()) {
      samples.push({ at: now, bytes: cumulativeBytes });
      while (samples.length > 2 && now - samples[0].at > windowMs) samples.shift();
    },
    speed(now = performance.now()) {
      if (samples.length < 2) return 0;
      const first = samples[0];
      const last = samples[samples.length - 1];
      // A stalled transfer stops calling `record`, so extend the window to now: the reading has to
      // fall towards zero while nothing arrives instead of freezing at the last good value.
      const elapsed = Math.max(last.at - first.at, now - first.at) / 1000;
      return elapsed > 0.05 ? Math.max(0, Math.round((last.bytes - first.bytes) / elapsed)) : 0;
    },
  };
}

// Bytes read before the throughput clock starts. TCP slow start, TLS and a CDN's first-byte work all
// land in the opening tens of kilobytes, so timing them measures the handshake rather than the link
// — which is how a nearby mirror loses a benchmark to a distant one that happened to answer first.
const BENCHMARK_WARMUP_BYTES = 32 * 1024;

// Below this the steady window is too short to divide by: one chunk that happens to carry both the
// warm-up and the rest would otherwise report an arbitrarily large rate.
const BENCHMARK_MINIMUM_STEADY_MS = 5;

async function readSample(response, { maximumBytes, warmupBytes, durationMs }) {
  if (!response.body) return { bytes: 0, steadyBytes: 0, steadyMs: 0 };
  const reader = response.body.getReader();
  const startedAt = performance.now();
  let bytes = 0;
  let steadyAt = 0;
  try {
    while (bytes < maximumBytes && performance.now() - startedAt < durationMs) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (!steadyAt && bytes >= warmupBytes) steadyAt = performance.now();
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  const finishedAt = performance.now();
  // The warm-up is charged at exactly `warmupBytes`, not at whatever the crossing chunk happened to
  // contain: attributing a whole large chunk to the handshake leaves nothing in the steady window.
  const steadyMs = steadyAt ? finishedAt - steadyAt : 0;
  return steadyAt && steadyMs >= BENCHMARK_MINIMUM_STEADY_MS
    ? { bytes, steadyBytes: bytes - warmupBytes, steadyMs }
    // Too small, or too fast, to separate the two — the whole sample is the only reading available.
    : { bytes, steadyBytes: bytes, steadyMs: finishedAt - startedAt };
}

/** Predicted seconds to fetch `sizeBytes` over this route.
 *
 * Ranking by the sample alone answers "which mirror replied first", which is not the question for a
 * multi-gigabyte wheel: there, one round trip of latency is noise and a 30% bandwidth difference is
 * minutes. A route that cannot serve ranges carries a real penalty rather than a cosmetic one — it
 * cannot be segmented and cannot be resumed, so an interruption costs the whole transfer again.
 */
function predictedSeconds(route, sizeBytes) {
  if (!route.ok) return Number.POSITIVE_INFINITY;
  const bytes = sizeBytes || route.totalBytes || 0;
  const transfer = bytes && route.speedBps ? bytes / route.speedBps : 0;
  return (route.latencyMs / 1000 + transfer) * (route.supportsRanges ? 1 : 1.25);
}

export async function benchmarkRoutes(routes, {
  fetcher = fetch,
  sampleBytes = 2 * 1024 ** 2,
  sampleMs = 1500,
  warmupBytes = BENCHMARK_WARMUP_BYTES,
  timeoutMs = 10000,
  sizeHint = 0,
} = {}) {
  const results = await Promise.all(routes.map(async (route) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const startedAt = performance.now();
    try {
      const response = await fetcher(route.url, {
        headers: { ...route.headers, Range: `bytes=0-${sampleBytes - 1}`, "User-Agent": userAgent },
        redirect: "follow",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const headersAt = performance.now();
      const range = contentRange(response.headers.get("content-range"));
      const totalBytes = range?.total || Number(response.headers.get("content-length") || 0);
      // Sampling a fixed 2 MB to choose a mirror for a 6 MB model would spend more on the probe than
      // on the file. Scale the sample to what is actually being fetched, with a floor that still
      // clears the warm-up.
      const target = sizeHint || totalBytes;
      const maximumBytes = target
        ? Math.max(warmupBytes + 64 * 1024, Math.min(sampleBytes, Math.ceil(target / 20)))
        : sampleBytes;
      const sample = await readSample(response, { maximumBytes, warmupBytes, durationMs: sampleMs });
      if (!sample.bytes) throw new Error("服务器没有返回数据");
      const elapsedSeconds = Math.max(sample.steadyMs / 1000, 0.001);
      return {
        ...route,
        ok: true,
        latencyMs: Math.round(headersAt - startedAt),
        speedBps: Math.round(sample.steadyBytes / elapsedSeconds),
        sampledBytes: sample.bytes,
        totalBytes,
        supportsRanges: response.status === 206 && Boolean(range?.total),
      };
    } catch (error) {
      return { ...route, ok: false, error: error.name === "AbortError" ? "测速超时" : error.message };
    } finally {
      clearTimeout(timeout);
    }
  }));

  return results.sort((first, second) => {
    if (first.ok !== second.ok) return first.ok ? -1 : 1;
    if (!first.ok) return 0;
    return predictedSeconds(first, sizeHint) - predictedSeconds(second, sizeHint);
  });
}

async function sha256(filePath) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  return hash.digest("hex");
}

async function writeAt(file, buffer, position) {
  let offset = 0;
  while (offset < buffer.length) {
    const { bytesWritten } = await file.write(buffer, offset, buffer.length - offset, position + offset);
    if (!bytesWritten) throw new Error("无法写入下载文件");
    offset += bytesWritten;
  }
}

// How many times a segment may fail *without moving* on one route. A stall that still wrote bytes is
// progress, not a failure, so it does not spend the budget — otherwise four slow patches anywhere in
// a multi-gigabyte download end the whole transfer even though it never stopped advancing.
const IDLE_ATTEMPT_LIMIT = 4;
// A ceiling on the retries a link that advances a little and dies can buy, so a pathological route
// cannot loop forever instead of falling through to the next one.
const TOTAL_ATTEMPT_LIMIT = 64;
// Hard failures a route may accumulate across the whole download before it is dropped. Without this
// a single dead mirror would be re-leased by every segment in turn.
const ROUTE_FAILURE_LIMIT = 3;
// A segment smaller than this is not worth splitting: the second request's handshake would cost
// more than the halved transfer saves. Moving a segment is judged on time rather than size — see
// `shouldLeave` — because the last megabyte of a file on a mirror that has collapsed to a few KB/s
// is exactly the case where a fresh request pays for itself many times over.
const MINIMUM_SPLIT_BYTES = 4 * 1024 ** 2;
// What a move costs before it transfers anything: DNS, connect, TLS and the first byte. Measured
// TTFB against the configured mirrors sits well under a second; this leaves room for the worst of
// them without ever making a genuinely useful move look unprofitable.
const ROUTE_SWITCH_HANDSHAKE_SECONDS = 1.5;
// How often an in-flight segment re-examines whether its mirror is still the right one.
const ROUTE_REVIEW_INTERVAL_MS = 4000;
// How many times one segment may move to a different mirror, so a pair of routes that measure
// alternately faster cannot trade a segment back and forth instead of downloading it.
const ROUTE_SWITCH_LIMIT = 3;
// A mirror is only abandoned mid-segment when another one is this many times faster per connection,
// and only when the move can still pay for itself.
const SLOW_ROUTE_RATIO = 2.5;
const SLOW_ROUTE_MINIMUM_SECONDS = 15;

/** The mirrors a download may draw from, and who is currently using them.
 *
 * Ranking once and then committing the whole file to the winner is what makes a download "stick" to
 * one source: the sample is 2 MB taken before the transfer, and the mirror that wins it is regularly
 * not the one that sustains the best rate for the next twenty minutes. So routes stay a live pool —
 * segments are spread across them from the start, throughput is measured per route while the
 * transfer runs, and a segment can leave a route that turned out to be slow.
 */
function createRoutePool(routes, connections, { now = () => performance.now() } = {}) {
  const best = Math.max(...routes.map((route) => route.speedBps || 0), 0);
  // A mirror an order of magnitude slower than the best one is not a useful second source; handing
  // it an equal share of the segments would make it the tail that decides when the file is done.
  const eligible = routes.filter((route) => !best || (route.speedBps || 0) >= best * 0.2);
  const pool = (eligible.length ? eligible : routes).map((route) => ({
    route,
    active: 0,
    failures: 0,
    disabled: false,
    bytes: 0,
    meter: createSpeedometer(),
  }));
  const capacity = Math.max(1, Math.ceil(connections / pool.length));
  const entryOf = (route) => pool.find((entry) => entry.route === route);

  // Per connection, so a route running four segments is compared with an idle one on the same
  // terms: the idle route's benchmark reading was taken over a single connection too.
  const laneSpeed = (entry) => {
    if (!entry.active) return entry.route.speedBps || 0;
    const measured = entry.meter.speed(now());
    return measured ? measured / entry.active : (entry.route.speedBps || 0);
  };

  return {
    routes: pool.map((entry) => entry.route),
    size: pool.length,
    lease(excluded = new Set()) {
      const live = pool.filter((entry) => !entry.disabled);
      if (!live.length) return null;
      const preferred = live.filter((entry) => !excluded.has(entry.route.url));
      const candidates = preferred.length ? preferred : live;
      const free = candidates.filter((entry) => entry.active < capacity);
      const choices = free.length ? free : candidates;
      return choices.sort((first, second) => laneSpeed(second) - laneSpeed(first))[0].route;
    },
    acquire(route) {
      const entry = entryOf(route);
      if (entry) entry.active += 1;
    },
    release(route) {
      const entry = entryOf(route);
      if (entry) entry.active = Math.max(0, entry.active - 1);
    },
    record(route, bytes) {
      const entry = entryOf(route);
      if (!entry) return;
      entry.bytes += bytes;
      entry.meter.record(entry.bytes, now());
    },
    penalise(route) {
      const entry = entryOf(route);
      if (!entry) return;
      entry.failures += 1;
      if (entry.failures >= ROUTE_FAILURE_LIMIT) entry.disabled = true;
    },
    /** Whether this segment should abandon `route` for a materially faster one. */
    shouldLeave(route, segment) {
      if (segment.switches >= ROUTE_SWITCH_LIMIT) return false;
      const entry = entryOf(route);
      if (!entry) return false;
      const alternatives = pool.filter((candidate) => !candidate.disabled && candidate !== entry);
      if (!alternatives.length) return false;
      const mine = laneSpeed(entry);
      if (!mine) return false;
      const remaining = segment.end - segment.start - segment.downloaded + 1;
      // Deliberately not gated on `MINIMUM_SPLIT_BYTES`. The last segment of a download is the one
      // that decides when the file is finished, and when its mirror has collapsed the remainder is
      // small but the wait is not: a megabyte at 9 KB/s is nearly two minutes while every other
      // route sits idle. Size is the wrong question; the two tests below ask the right one.
      const stayingSeconds = remaining / mine;
      if (stayingSeconds < SLOW_ROUTE_MINIMUM_SECONDS) return false;
      const bestAlternative = Math.max(...alternatives.map(laneSpeed));
      if (bestAlternative < mine * SLOW_ROUTE_RATIO) return false;
      // Charge the move its handshake, so a switch is only taken when finishing somewhere else is
      // still faster than staying even after paying to get there.
      if (ROUTE_SWITCH_HANDSHAKE_SECONDS + remaining / bestAlternative >= stayingSeconds) return false;
      segment.switches += 1;
      return true;
    },
    snapshot() {
      return pool
        .filter((entry) => entry.bytes > 0 || entry.active > 0)
        .map((entry) => ({
          id: entry.route.id,
          label: entry.route.label,
          connections: entry.active,
          bytes: entry.bytes,
          speedBps: entry.meter.speed(now()),
        }));
    },
  };
}

/** Contiguous ranges covering the whole file, one per connection. */
function planSegments(totalBytes, connections) {
  const segments = [];
  const segmentSize = Math.ceil(totalBytes / connections);
  for (let index = 0; index < connections; index += 1) {
    const start = index * segmentSize;
    if (start >= totalBytes) break;
    segments.push({
      start,
      end: Math.min(totalBytes - 1, start + segmentSize - 1),
      downloaded: 0,
      switches: 0,
    });
  }
  return segments;
}

/** Hand the back half of a segment that is still running to an idle connection.
 *
 * Equal ranges finish at wildly unequal times once they are spread over mirrors of different speeds,
 * and the file is not done until the slowest one is. Splitting the laggard's remainder is what turns
 * "one slow mirror decides the finish time" into "the fast mirrors absorb its tail" — and it is the
 * mechanism that makes a mixed-speed set of sources behave like one fast source.
 */
function splitSegment(segment) {
  const position = segment.start + segment.downloaded;
  const remaining = segment.end - position + 1;
  if (remaining < MINIMUM_SPLIT_BYTES) return null;
  const boundary = position + Math.floor(remaining / 2) - 1;
  const stolen = { start: boundary + 1, end: segment.end, downloaded: 0, switches: 0 };
  // The running writer re-reads `end` on every chunk, so shrinking it here makes that request stop
  // at the new boundary and exit cleanly rather than writing over the thief's range.
  segment.end = boundary;
  return stolen;
}

async function fetchSegmentFromRoute({ fetcher, route, file, segment, pool, report, timeoutMs, cancelled }) {
  let idleAttempts = 0;
  let lastError = null;
  for (let attempt = 1; segment.start + segment.downloaded <= segment.end && attempt <= TOTAL_ATTEMPT_LIMIT; attempt += 1) {
    if (cancelled()) return;
    const downloadedBefore = segment.downloaded;
    const controller = new AbortController();
    let switching = false;
    let timeout;
    let review;
    const resetTimeout = () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => controller.abort(), timeoutMs);
    };
    resetTimeout();
    try {
      let position = segment.start + segment.downloaded;
      const response = await fetcher(route.url, {
        headers: { ...route.headers, Range: `bytes=${position}-${segment.end}`, "User-Agent": userAgent },
        redirect: "follow",
        signal: controller.signal,
      });
      const range = contentRange(response.headers.get("content-range"));
      if (response.status !== 206 || range?.start !== position) {
        throw new Error(`服务器没有返回请求的分片（HTTP ${response.status}）`);
      }
      review = setInterval(() => {
        if (cancelled()) {
          controller.abort();
          return;
        }
        if (!pool.shouldLeave(route, segment)) return;
        switching = true;
        controller.abort();
      }, ROUTE_REVIEW_INTERVAL_MS);
      for await (const value of response.body) {
        resetTimeout();
        const chunk = Buffer.from(value);
        const remaining = segment.end - position + 1;
        if (remaining <= 0) break;
        const writable = chunk.length > remaining ? chunk.subarray(0, remaining) : chunk;
        await writeAt(file, writable, position);
        position += writable.length;
        segment.downloaded += writable.length;
        pool.record(route, writable.length);
        report();
        if (position > segment.end) break;
      }
      if (segment.start + segment.downloaded <= segment.end) throw new Error("分片提前结束");
      return;
    } catch (error) {
      lastError = error;
      if (cancelled()) return;
      if (switching) throw Object.assign(new Error(`${route.label} 速度明显落后，分片改用更快的线路`), { switchRoute: true });
      idleAttempts = segment.downloaded > downloadedBefore ? 0 : idleAttempts + 1;
      if (idleAttempts >= IDLE_ATTEMPT_LIMIT || attempt >= TOTAL_ATTEMPT_LIMIT) throw error;
    } finally {
      clearTimeout(timeout);
      clearInterval(review);
    }
  }
  if (segment.start + segment.downloaded <= segment.end && !cancelled()) {
    throw lastError || new Error("分片未完成");
  }
}

async function runSegment(segment, context) {
  const excluded = new Set();
  let lastError;
  while (!context.cancelled()) {
    const route = context.pool.lease(excluded);
    if (!route) throw lastError || new Error("没有可用的下载线路");
    context.pool.acquire(route);
    try {
      await fetchSegmentFromRoute({ ...context, route, segment });
      return;
    } catch (error) {
      lastError = error;
      excluded.add(route.url);
      if (error.switchRoute) {
        context.onWarning?.(error.message);
      } else {
        context.pool.penalise(route);
        context.onWarning?.(`${route.label} 分片失败（${error.message}），正在改用其他线路继续`);
      }
    } finally {
      context.pool.release(route);
    }
  }
}

const METADATA_VERSION = 3;

async function readSegmentMetadata(metadataPath, temporaryPath, totalBytes) {
  if (!existsSync(metadataPath) || !existsSync(temporaryPath)) return null;
  try {
    const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
    if (metadata.version !== METADATA_VERSION || metadata.totalBytes !== totalBytes) return null;
    if ((await stat(temporaryPath)).size !== totalBytes) return null;
    const segments = metadata.segments;
    if (!Array.isArray(segments) || !segments.length) return null;
    let expectedStart = 0;
    for (const segment of segments) {
      if (!Number.isInteger(segment.start) || !Number.isInteger(segment.end)) return null;
      if (segment.start !== expectedStart || segment.end < segment.start) return null;
      if (!Number.isInteger(segment.downloaded) || segment.downloaded < 0) return null;
      if (segment.downloaded > segment.end - segment.start + 1) return null;
      expectedStart = segment.end + 1;
    }
    // A partition that does not cover the file exactly would silently leave a hole in it.
    if (expectedStart !== totalBytes) return null;
    return segments.map((segment) => ({ ...segment, switches: 0 }));
  } catch {
    return null;
  }
}

// The previous layout kept one `.part.N` file per connection and merged them at the end. Files
// written by that version are unusable here and would otherwise sit in the cache directory forever.
async function clearLegacyParts(temporaryPath, connections) {
  let count = Math.max(connections, 16);
  try {
    const metadata = JSON.parse(await readFile(`${temporaryPath}.json`, "utf8"));
    count = Math.max(count, Number(metadata.connections || 0));
  } catch {}
  await Promise.all(Array.from({ length: count }, (_, index) => rm(`${temporaryPath}.${index}`, { force: true })));
}

async function clearDownloadParts(temporaryPath, connections) {
  await clearLegacyParts(temporaryPath, connections);
  await rm(`${temporaryPath}.json`, { force: true });
}

/** Fetch one file from every usable mirror at once.
 *
 * Segments are dealt across the ranked routes instead of all going to the winner of the benchmark,
 * each segment can leave a route that turns out to be slow, and a connection with nothing left to do
 * takes over half of whatever segment is furthest behind. Bytes land directly in the output file at
 * their absolute offsets, so finishing costs no merge pass and no second copy of a multi-gigabyte
 * wheel on disk.
 */
async function downloadParallel({
  fetcher, routes, temporaryPath, totalBytes, connections, onProgress, onWarning, timeoutMs, maximumBytes,
}) {
  if (maximumBytes && totalBytes > maximumBytes) throw new Error("下载文件超过允许的最大大小");
  const metadataPath = `${temporaryPath}.json`;
  const restored = await readSegmentMetadata(metadataPath, temporaryPath, totalBytes);
  if (!restored) await clearLegacyParts(temporaryPath, connections);
  const segments = restored || planSegments(totalBytes, connections);
  const resumedBytes = segments.reduce((sum, segment) => sum + segment.downloaded, 0);

  const file = await open(temporaryPath, existsSync(temporaryPath) ? "r+" : "w+");
  // A partial file left by an older engine, or by a mirror that was serving something else, is
  // still sitting at whatever length it reached. Every byte of it is about to be rewritten, but a
  // leftover tail past the end would survive into the finished file.
  if (!restored) await file.truncate(totalBytes);
  const pool = createRoutePool(routes, connections);
  const meter = createSpeedometer();
  const startedAt = performance.now();
  let failure = null;
  let lastReportedAt = 0;
  let lastPersistedAt = performance.now();
  let persisting = false;

  const currentBytes = () => segments.reduce((sum, segment) => sum + segment.downloaded, 0);
  const cancelled = () => failure != null;

  const persist = async () => {
    if (persisting) return;
    persisting = true;
    try {
      // Recording more progress than is actually on disk would resume past a hole, so the data is
      // flushed before the bookkeeping that claims it exists.
      await file.sync();
      await writeFile(metadataPath, JSON.stringify({
        version: METADATA_VERSION,
        totalBytes,
        segments: segments.map(({ start, end, downloaded }) => ({ start, end, downloaded })),
      }), "utf8");
    } catch {
      // Losing the resume record only costs a re-download; it must never end a live transfer.
    } finally {
      persisting = false;
    }
  };

  const report = (force = false) => {
    const now = performance.now();
    if (!force && now - lastReportedAt < 120) return;
    lastReportedAt = now;
    const current = currentBytes();
    meter.record(current, now);
    const speedBps = meter.speed(now);
    const elapsedSeconds = Math.max((now - startedAt) / 1000, 0.001);
    const sources = pool.snapshot();
    onProgress?.({
      currentBytes: current,
      totalBytes,
      speedBps,
      // The lifetime rate over this run only, so a resumed transfer does not claim the bytes it
      // found on disk as if it had just fetched them.
      averageBps: Math.round((current - resumedBytes) / elapsedSeconds),
      etaSeconds: speedBps ? Math.round((totalBytes - current) / speedBps) : 0,
      connections: sources.reduce((sum, source) => sum + source.connections, 0) || segments.length,
      sources,
      resumed: resumedBytes > 0,
    });
  };

  await persist();
  report(true);
  const ticker = setInterval(() => {
    report(true);
    const now = performance.now();
    if (now - lastPersistedAt >= 5000) {
      lastPersistedAt = now;
      void persist();
    }
  }, 500);

  const pending = segments.filter((segment) => segment.start + segment.downloaded <= segment.end);
  const takeSegment = () => {
    const next = pending.shift();
    if (next) return next;
    const laggard = segments
      .filter((segment) => segment.start + segment.downloaded <= segment.end)
      .sort((first, second) => (second.end - second.start - second.downloaded) - (first.end - first.start - first.downloaded))[0];
    if (!laggard) return null;
    const stolen = splitSegment(laggard);
    if (stolen) segments.push(stolen);
    return stolen;
  };

  const context = { fetcher, file, pool, report, timeoutMs, onWarning, cancelled };
  const workerCount = Math.max(1, Math.min(connections, segments.length));
  try {
    await Promise.all(Array.from({ length: workerCount }, async () => {
      while (!cancelled()) {
        const segment = takeSegment();
        if (!segment) return;
        try {
          await runSegment(segment, context);
        } catch (error) {
          failure ||= error;
          return;
        }
      }
    }));
    if (failure) throw failure;
    const written = currentBytes();
    if (written !== totalBytes) throw new Error(`下载不完整：${written}/${totalBytes}`);
    await file.sync();
  } finally {
    clearInterval(ticker);
    await persist();
    await file.close();
  }
  report(true);
}

// The parallel path retries each segment on every mirror; this one is for routes that cannot serve
// ranges, where a retry restarts from zero and the only thing left to do is try the next route.
async function downloadSingle(fetcher, route, temporaryPath, onProgress, timeoutMs, maximumBytes) {
  const partSize = async () => (existsSync(temporaryPath) ? (await stat(temporaryPath)).size : 0);
  let idleAttempts = 0;
  for (let attempt = 1; attempt <= TOTAL_ATTEMPT_LIMIT; attempt += 1) {
    const downloadedBefore = await partSize();
    try {
      await downloadSingleAttempt(fetcher, route, temporaryPath, onProgress, timeoutMs, maximumBytes);
      return;
    } catch (error) {
      // Without ranges a retry restarts from zero, which is not a retry — it is the same download
      // again. Fall through to the next route instead of spending the link on it.
      if (!route.supportsRanges) throw error;
      idleAttempts = (await partSize()) > downloadedBefore ? 0 : idleAttempts + 1;
      if (idleAttempts >= IDLE_ATTEMPT_LIMIT) throw error;
    }
  }
}

async function downloadSingleAttempt(fetcher, route, temporaryPath, onProgress, timeoutMs, maximumBytes) {
  const meter = createSpeedometer();
  const startedAt = performance.now();
  let currentBytes = existsSync(temporaryPath) ? (await stat(temporaryPath)).size : 0;
  if (!route.supportsRanges || currentBytes > route.totalBytes) currentBytes = 0;
  const resumedBytes = currentBytes;
  let lastReportedAt = 0;
  if (route.totalBytes && currentBytes === route.totalBytes) {
    onProgress?.({ currentBytes, totalBytes: route.totalBytes, speedBps: 0, connections: 1, resumed: true });
    return;
  }
  // Armed only once there is a request to time out. Arming it above the early return left a
  // five-minute timer behind on an already finished download, which is enough to hold the whole
  // process open long after its work is done.
  const controller = new AbortController();
  let timeout;
  const resetTimeout = () => {
    clearTimeout(timeout);
    timeout = setTimeout(() => controller.abort(), timeoutMs);
  };
  resetTimeout();
  let response;
  let file;
  try {
    response = await fetcher(route.url, {
      headers: {
        ...route.headers,
        ...(currentBytes ? { Range: `bytes=${currentBytes}-` } : {}),
        "User-Agent": userAgent,
      },
      redirect: "follow",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const range = contentRange(response.headers.get("content-range"));
    if (currentBytes && (response.status !== 206 || range?.start !== currentBytes)) {
      await response.body?.cancel().catch(() => {});
      currentBytes = 0;
      response = await fetcher(route.url, {
        headers: { ...route.headers, "User-Agent": userAgent },
        redirect: "follow",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }
    const responseRange = contentRange(response.headers.get("content-range"));
    const totalBytes = responseRange?.total || route.totalBytes || currentBytes + Number(response.headers.get("content-length") || 0);
    if (maximumBytes && totalBytes > maximumBytes) throw new Error("下载文件超过允许的最大大小");
    file = await open(temporaryPath, currentBytes ? "r+" : "w");
    meter.record(currentBytes);
    onProgress?.({ currentBytes, totalBytes, speedBps: 0, connections: 1, resumed: currentBytes > 0 });
    for await (const value of response.body) {
      resetTimeout();
      const chunk = Buffer.from(value);
      await writeAt(file, chunk, currentBytes);
      currentBytes += chunk.length;
      if (maximumBytes && currentBytes > maximumBytes) throw new Error("下载文件超过允许的最大大小");
      const now = performance.now();
      if (now - lastReportedAt >= 120) {
        lastReportedAt = now;
        meter.record(currentBytes, now);
        const speedBps = meter.speed(now);
        onProgress?.({
          currentBytes,
          totalBytes,
          speedBps,
          averageBps: Math.round((currentBytes - resumedBytes) / Math.max((now - startedAt) / 1000, 0.001)),
          etaSeconds: speedBps && totalBytes ? Math.round((totalBytes - currentBytes) / speedBps) : 0,
          connections: 1,
        });
      }
    }
    if (totalBytes && currentBytes !== totalBytes) throw new Error(`文件不完整：${currentBytes}/${totalBytes}`);
    const now = performance.now();
    meter.record(currentBytes, now);
    onProgress?.({ currentBytes, totalBytes, speedBps: meter.speed(now), connections: 1, etaSeconds: 0 });
    await file.sync();
  } finally {
    clearTimeout(timeout);
    await file?.close();
  }
}

/** Routes that agree with the leader about the file, and can therefore serve part of it.
 *
 * A mirror carrying a different build under the same name reports a different length. Splitting one
 * download across both would interleave two files and produce a checksum failure at best; dropping
 * the disagreeing mirror keeps it available as a whole-file fallback instead.
 */
function consistentRangedRoutes(available) {
  const ranged = available.filter((route) => route.supportsRanges && route.totalBytes);
  if (ranged.length < 2) return ranged;
  const expected = ranged[0].totalBytes;
  return ranged.filter((route) => route.totalBytes === expected);
}

export async function downloadFile({
  routes,
  destination,
  expectedSha256,
  connections = 8,
  thresholdBytes = 64 * 1024 ** 2,
  // An *idle* timeout: it restarts on every chunk, so it only fires when a connection has gone
  // quiet. Mirrors carrying multi-gigabyte model files routinely go quiet for minutes under load,
  // and aborting one that is merely slow costs the whole route.
  timeoutMs = 5 * 60 * 1000,
  fetcher = fetch,
  rankRoutes = true,
  sizeHint = 0,
  onRoute,
  onProgress,
  onWarning,
  maximumBytes = 0,
  existingFilePolicy = "reuse",
}) {
  await mkdir(path.dirname(destination), { recursive: true });
  const normalizedDigest = expectedSha256?.toLowerCase();
  if (existsSync(destination)) {
    if ((!normalizedDigest && existingFilePolicy === "reuse") || (normalizedDigest && await sha256(destination) === normalizedDigest)) {
      const totalBytes = (await stat(destination)).size;
      onRoute?.({ id: "cache", label: "本地缓存", cached: true, connections: 0 });
      onProgress?.({ currentBytes: totalBytes, totalBytes, speedBps: 0, connections: 0, cached: true });
      return { path: destination, cached: true, route: null, connections: 0 };
    }
    if (existingFilePolicy === "error") throw new Error("目标目录中已存在同名文件，且无法验证其来源");
  }
  if (existsSync(destination)) await rm(destination, { force: true });

  const ranked = await benchmarkRoutes(routes, { fetcher, sizeHint });
  const withinLimit = (route) => !maximumBytes || !route.totalBytes || route.totalBytes <= maximumBytes;
  const available = rankRoutes
    ? ranked.filter((route) => route.ok && withinLimit(route))
    : routes.map((route) => ranked.find((result) => result.url === route.url)).filter((route) => route?.ok && withinLimit(route));
  if (!available.length) {
    throw new Error(`所有下载线路均不可用：${ranked.map((route) => `${route.label}: ${route.error}`).join("；")}`);
  }

  const temporaryPath = `${destination}.part`;
  const finish = async (route, activeConnections, sources) => {
    if (normalizedDigest) {
      const digest = await sha256(temporaryPath);
      if (digest !== normalizedDigest) {
        await rm(temporaryPath, { force: true });
        await clearDownloadParts(temporaryPath, activeConnections);
        throw new Error("SHA-256 校验不匹配");
      }
    }
    await rm(destination, { force: true });
    await rename(temporaryPath, destination);
    await clearDownloadParts(temporaryPath, activeConnections);
    return { path: destination, cached: false, route, connections: activeConnections, sources };
  };

  // Every mirror that agrees about the file works on it together, so one slow source costs a share
  // of the transfer rather than all of it.
  const parallelRoutes = consistentRangedRoutes(available);
  const leader = parallelRoutes[0];
  const parallelConnections = leader ? partCount(leader.totalBytes, connections, thresholdBytes) : 1;
  let fallbackRoutes = available;
  if (leader && parallelConnections > 1) {
    const pooled = parallelRoutes.slice(0, Math.max(1, Math.min(parallelRoutes.length, connections)));
    onRoute?.({
      id: leader.id,
      label: pooled.length > 1 ? `${leader.label} 等 ${pooled.length} 条线路` : leader.label,
      latencyMs: leader.latencyMs,
      speedBps: leader.speedBps,
      totalBytes: leader.totalBytes,
      supportsRanges: true,
      connections: parallelConnections,
      sources: pooled.map((route) => ({ id: route.id, label: route.label, speedBps: route.speedBps })),
      attempt: 1,
    });
    try {
      let sources = [];
      await downloadParallel({
        fetcher,
        routes: pooled,
        temporaryPath,
        totalBytes: leader.totalBytes,
        connections: parallelConnections,
        onProgress: (event) => {
          sources = event.sources || sources;
          onProgress?.({ ...event, route: event.sources?.length > 1 ? `${pooled.length} 条线路` : leader.label });
        },
        onWarning,
        timeoutMs,
        maximumBytes,
      });
      return await finish(leader, parallelConnections, sources);
    } catch (error) {
      // Every route in the pool has already been tried on every segment, and the partial file
      // belongs to them. What is left is the routes the pool would not accept — a mirror without
      // range support, or one serving a different length under the same name — and their bytes do
      // not belong in this `.part` file, so it goes before they are used. With none of those, the
      // failure is the download's: the partial file and its resume record stay for the next run.
      fallbackRoutes = available.filter((route) => !parallelRoutes.includes(route));
      if (!fallbackRoutes.length) throw error;
      onWarning?.(`多线路下载失败（${error.message}），正在改用其余线路重新下载`);
      await rm(temporaryPath, { force: true });
      await clearDownloadParts(temporaryPath, parallelConnections);
    }
  }

  let lastError;
  for (let index = 0; index < fallbackRoutes.length; index += 1) {
    const route = fallbackRoutes[index];
    onRoute?.({
      id: route.id,
      label: route.label,
      latencyMs: route.latencyMs,
      speedBps: route.speedBps,
      totalBytes: route.totalBytes,
      supportsRanges: route.supportsRanges,
      connections: 1,
      attempt: index + 1,
    });
    try {
      await downloadSingle(fetcher, route, temporaryPath, (event) => onProgress?.({ ...event, route: route.label }), timeoutMs, maximumBytes);
      return await finish(route, 1, []);
    } catch (error) {
      lastError = error;
      if (index + 1 < fallbackRoutes.length) onWarning?.(`${route.label} 下载失败（${error.message}），正在切换线路`);
    }
  }
  throw lastError || new Error("下载失败");
}

export const downloadInternals = {
  contentRange,
  partCount,
  planSegments,
  splitSegment,
  predictedSeconds,
  consistentRangedRoutes,
  createSpeedometer,
  createRoutePool,
};
