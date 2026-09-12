export const VIEWER_TOOLS = Object.freeze(["move", "brush", "eraser"]);
export const VIEWER_EDGE_STYLES = Object.freeze(["solid", "dashed", "dotted", "double", "glow"]);
export const VIEWER_DEFAULT_COLOR = "#c8acfb";
export const VIEWER_MAX_STROKE_POINTS = 10_000;
export const VIEWER_MAX_LAYOUT_POINTS = 50_000;
export const VIEWER_MAX_LAYOUT_STROKES = 1_000;
export const VIEWER_MAX_LAYOUT_BYTES = 1024 * 1024;
export const VIEWER_MAX_CANVAS_EDGE = 24_576;
export const MAX_EXPORT_PIXELS = 64 * 1024 * 1024;
export const VIEWER_MAX_FILE_BYTES = 128 * 1024 * 1024;
export const VIEWER_MAX_BATCH_BYTES = 256 * 1024 * 1024;
export const VIEWER_MAX_LAYER_SOURCE_BYTES = 256 * 1024 * 1024;
export const VIEWER_MAX_ANIMATED_SOURCE_BYTES = 32 * 1024 * 1024;
export const VIEWER_MAX_ANIMATED_TOTAL_BYTES = 128 * 1024 * 1024;
export const VIEWER_MAX_OUTPUT_BYTES = 128 * 1024 * 1024;
export const VIEWER_MAX_RESTORE_PIXELS = 128 * 1024 * 1024;
export const VIEWER_MAX_UNDO_STEPS = 50;
export const VIEWER_MAX_UNDO_POINTS = 200_000;
export const VIEWER_MAX_UNDO_BYTES = 32 * 1024 * 1024;
export const VIEWER_MAX_UNDO_SOURCE_BYTES = 256 * 1024 * 1024;
export const VIEWER_RESIZE_HANDLES = Object.freeze(["tl", "tr", "bl", "br", "top", "right", "bottom", "left"]);
// Which edges a handle moves: -1 pulls the left/top edge, 1 pushes the right/bottom one, 0 leaves
// that axis alone. Spelled out because the names overlap as substrings -- "right" contains "t".
export const VIEWER_RESIZE_HANDLE_AXES = Object.freeze({
  tl: { x: -1, y: -1 }, tr: { x: 1, y: -1 }, bl: { x: -1, y: 1 }, br: { x: 1, y: 1 },
  top: { x: 0, y: -1 }, bottom: { x: 0, y: 1 }, left: { x: -1, y: 0 }, right: { x: 1, y: 0 },
});

const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, finite(value, minimum)));

export function viewerAbortError(message = "操作已取消") {
  if (typeof DOMException === "function") return new DOMException(message, "AbortError");
  return Object.assign(new Error(message), { name: "AbortError" });
}

export function readViewerFileAsDataUrl(file, { signal, FileReaderClass = globalThis.FileReader } = {}) {
  const admission = viewerFileBatchAdmission([file]);
  if (!admission.ok) return Promise.reject(new Error(admission.reason));
  if (signal?.aborted) return Promise.reject(signal.reason?.name === "AbortError" ? signal.reason : viewerAbortError());
  return new Promise((resolve, reject) => {
    const reader = new FileReaderClass();
    let settled = false;
    const cleanup = () => signal?.removeEventListener?.("abort", onAbort);
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const onAbort = () => {
      try { reader.abort?.(); } catch {}
      finish(reject, signal.reason?.name === "AbortError" ? signal.reason : viewerAbortError());
    };
    reader.onload = () => finish(resolve, reader.result);
    reader.onerror = () => finish(reject, new Error("图片读取失败"));
    reader.onabort = () => finish(reject, viewerAbortError());
    signal?.addEventListener?.("abort", onAbort, { once: true });
    try { reader.readAsDataURL(file); } catch (error) { finish(reject, error); }
  });
}

export function viewerCanvasDimensions(width, height, purpose = "画布") {
  const normalizedWidth = Math.ceil(Number(width));
  const normalizedHeight = Math.ceil(Number(height));
  if (!Number.isFinite(normalizedWidth) || !Number.isFinite(normalizedHeight) || normalizedWidth < 1 || normalizedHeight < 1) {
    throw new Error(`${purpose}尺寸无效`);
  }
  if (normalizedWidth > VIEWER_MAX_CANVAS_EDGE || normalizedHeight > VIEWER_MAX_CANVAS_EDGE) {
    throw new Error(`${purpose}尺寸 ${normalizedWidth} × ${normalizedHeight} 超过 ${VIEWER_MAX_CANVAS_EDGE} 像素边长上限，请缩小图层后重试`);
  }
  if (normalizedWidth * normalizedHeight > MAX_EXPORT_PIXELS) {
    throw new Error(`${purpose}尺寸 ${normalizedWidth} × ${normalizedHeight} 超过 64 MP 安全面积上限，请缩小图层后重试`);
  }
  return { width: normalizedWidth, height: normalizedHeight, pixels: normalizedWidth * normalizedHeight };
}

export function assertViewerByteBudget(bytes, maximumBytes, purpose = "图片") {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) throw new Error(`${purpose}大小无效`);
  if (value > maximumBytes) throw new Error(`${purpose}超过 ${Math.round(maximumBytes / 1024 / 1024)} MiB 上限`);
  return value;
}

export function viewerStrokePointCount(layers) {
  let total = 0;
  for (const layer of Array.isArray(layers) ? layers : []) {
    if (!Array.isArray(layer?.paintStrokes)) continue;
    for (const stroke of layer.paintStrokes) total += Array.isArray(stroke?.points) ? stroke.points.length : 0;
  }
  return total;
}

export function viewerStrokeCount(layers) {
  let total = 0;
  for (const layer of Array.isArray(layers) ? layers : []) total += Array.isArray(layer?.paintStrokes) ? layer.paintStrokes.length : 0;
  return total;
}

export function viewerDataUrlBytes(source) {
  if (typeof source !== "string" || !source.startsWith("data:")) return 0;
  const comma = source.indexOf(",");
  if (comma < 0) return Infinity;
  const header = source.slice(0, comma);
  const payload = source.slice(comma + 1);
  if (!/;base64(?:;|$)/i.test(header)) return payload.length;
  const padding = payload.endsWith("==") ? 2 : payload.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor(payload.length * 3 / 4) - padding);
}

export function viewerAnimatedBatchAdmission(layers) {
  let totalBytes = 0;
  for (const layer of Array.isArray(layers) ? layers : []) {
    const source = layer?.url || layer?.originalUrl;
    const bytes = Number.isFinite(Number(layer?.sourceBytes)) ? Number(layer.sourceBytes) : viewerDataUrlBytes(source);
    if (!Number.isFinite(bytes) || bytes > VIEWER_MAX_ANIMATED_SOURCE_BYTES) {
      return { ok: false, totalBytes, reason: "GIF 拼图单个源图超过 32 MiB 上限" };
    }
    totalBytes += bytes;
    if (totalBytes > VIEWER_MAX_ANIMATED_TOTAL_BYTES) {
      return { ok: false, totalBytes, reason: "GIF 拼图图片源总量超过 128 MiB 上限" };
    }
  }
  return { ok: true, totalBytes, reason: "" };
}

export function viewerFileBatchAdmission(files, { maximumFileBytes = VIEWER_MAX_FILE_BYTES, maximumBatchBytes = VIEWER_MAX_BATCH_BYTES } = {}) {
  let totalBytes = 0;
  for (const file of Array.isArray(files) ? files : []) {
    const bytes = Number(file?.size);
    if (!Number.isFinite(bytes) || bytes < 0) return { ok: false, totalBytes, reason: "图片文件大小无效" };
    if (bytes > maximumFileBytes) return { ok: false, totalBytes: totalBytes + bytes, reason: `单张图片超过 ${Math.round(maximumFileBytes / 1024 / 1024)} MiB 上限` };
    totalBytes += bytes;
    if (totalBytes > maximumBatchBytes) return { ok: false, totalBytes, reason: `本批图片超过 ${Math.round(maximumBatchBytes / 1024 / 1024)} MiB 上限` };
  }
  return { ok: true, totalBytes, reason: "" };
}

export function viewerLayerSourceByteCount(layers) {
  let total = 0;
  for (const layer of Array.isArray(layers) ? layers : []) {
    const explicit = layer?.sourceBytes;
    const source = layer?.originalUrl || layer?.url;
    const bytes = typeof explicit === "number" && Number.isFinite(explicit) && explicit >= 0
      ? explicit
      : typeof source === "string" && source.startsWith("data:") ? viewerDataUrlBytes(source) : Infinity;
    if (!Number.isFinite(bytes)) return Infinity;
    total += bytes;
    if (!Number.isSafeInteger(total)) return Infinity;
  }
  return total;
}

export function viewerUndoSnapshotMetrics(snapshot) {
  const layers = Array.isArray(snapshot?.layers) ? snapshot.layers : [];
  let points = viewerStrokePointCount(layers);
  let structuralBytes = 128 + (Array.isArray(snapshot?.snappedLayers) ? snapshot.snappedLayers.length * 64 : 0);
  const sources = new Map();
  const addSource = (layer, source, explicitBytes) => {
    if (typeof source !== "string" || !source) return;
    const historyId = historyAssetIdFromUrl(source);
    const identity = historyId && historyId === layer?.assetId ? `history:${historyId}` : source;
    const estimated = Math.max(source.length, typeof explicitBytes === "number" && Number.isFinite(explicitBytes) && explicitBytes >= 0 ? explicitBytes : 0);
    sources.set(identity, Math.max(sources.get(identity) || 0, estimated));
  };
  const measureManualLayout = (layout) => {
    if (!layout || !Array.isArray(layout.layers)) return;
    if (layout.layers.length > 100) { structuralBytes = Infinity; return; }
    structuralBytes += 128;
    for (const nested of layout.layers) {
      if (!nested || typeof nested !== "object") continue;
      structuralBytes += 512;
      const nestedStrokes = Array.isArray(nested.paintStrokes) ? nested.paintStrokes : [];
      if (nestedStrokes.length > VIEWER_MAX_LAYOUT_STROKES) { structuralBytes = Infinity; return; }
      for (const stroke of nestedStrokes) {
        if (Array.isArray(stroke?.points) && stroke.points.length > VIEWER_MAX_STROKE_POINTS) { structuralBytes = Infinity; return; }
        const nestedPoints = Array.isArray(stroke?.points) ? stroke.points.length : 0;
        points += nestedPoints;
        structuralBytes += 192 + nestedPoints * 24;
      }
      addSource(nested, nested.originalUrl || nested.url, nested.sourceBytes);
      if (nested.url && nested.url !== nested.originalUrl) addSource(nested, nested.url, 0);
    }
  };
  for (const layer of layers) {
    structuralBytes += 512;
    for (const stroke of Array.isArray(layer?.paintStrokes) ? layer.paintStrokes : []) {
      structuralBytes += 192 + (Array.isArray(stroke?.points) ? stroke.points.length * 24 : 0);
    }
    addSource(layer, layer.originalUrl || layer.url, layer.sourceBytes);
    if (layer.url && layer.url !== layer.originalUrl) addSource(layer, layer.url, 0);
    measureManualLayout(layer?.manualLayout || layer?.manual_layout);
  }
  const sourceBytes = [...sources.values()].reduce((total, value) => total + value, 0);
  return { points, structuralBytes, sourceBytes, bytes: structuralBytes + sourceBytes, sources };
}

export function viewerUndoStackMetrics(stack) {
  let points = 0;
  let structuralBytes = 0;
  const sources = new Map();
  for (const snapshot of Array.isArray(stack) ? stack : []) {
    const metrics = viewerUndoSnapshotMetrics(snapshot);
    points += metrics.points;
    structuralBytes += metrics.structuralBytes;
    for (const [identity, bytes] of metrics.sources) sources.set(identity, Math.max(sources.get(identity) || 0, bytes));
  }
  const sourceBytes = [...sources.values()].reduce((total, value) => total + value, 0);
  return { points, structuralBytes, sourceBytes, bytes: structuralBytes + sourceBytes };
}

export function admitViewerUndoSnapshot(stack, snapshot, limits = {}) {
  const maximumSteps = limits.maximumSteps ?? VIEWER_MAX_UNDO_STEPS;
  const maximumPoints = limits.maximumPoints ?? VIEWER_MAX_UNDO_POINTS;
  const maximumBytes = limits.maximumBytes ?? VIEWER_MAX_UNDO_BYTES;
  const maximumSourceBytes = limits.maximumSourceBytes ?? VIEWER_MAX_UNDO_SOURCE_BYTES;
  const single = viewerUndoSnapshotMetrics(snapshot);
  if (single.points > maximumPoints || single.structuralBytes > maximumBytes || single.sourceBytes > maximumSourceBytes) {
    return { stack: Array.isArray(stack) ? stack : [], saved: false, reason: "当前画布单步撤销数据超过安全预算，本次操作不会加入撤销历史", metrics: single };
  }
  const candidates = [...(Array.isArray(stack) ? stack : []), snapshot].slice(-maximumSteps);
  let metrics = viewerUndoStackMetrics(candidates);
  while (candidates.length > 1 && (metrics.points > maximumPoints || metrics.structuralBytes > maximumBytes || metrics.sourceBytes > maximumSourceBytes)) {
    candidates.shift();
    metrics = viewerUndoStackMetrics(candidates);
  }
  return { stack: candidates, saved: true, reason: "", metrics };
}

export function boundedViewerUndoStack(stack, snapshot, limits = {}) {
  return admitViewerUndoSnapshot(stack, snapshot, limits).stack;
}

export function historyAssetIdFromUrl(value) {
  const match = typeof value === "string" ? value.match(/^\/api\/inference\/history\/assets\/([A-Za-z0-9_-]+)$/) : null;
  return match?.[1] || "";
}

export function viewerHistorySourcesBound(layer) {
  const assetId = typeof layer?.assetId === "string" ? layer.assetId : "";
  const sources = [layer?.url, layer?.originalUrl].filter((source) => typeof source === "string" && source.length);
  return Boolean(assetId && sources.length && sources.every((source) => historyAssetIdFromUrl(source) === assetId));
}

export async function mapViewerConcurrent(items, limit, worker, { signal: parentSignal, controller: suppliedController } = {}) {
  const values = Array.isArray(items) ? items : [];
  const results = new Array(values.length);
  let cursor = 0;
  let stopped = false;
  let firstError = null;
  const controller = suppliedController || (typeof AbortController === "function" ? new AbortController() : null);
  const signal = controller?.signal || parentSignal;
  const onParentAbort = () => controller?.abort(parentSignal?.reason || viewerAbortError());
  if (parentSignal && parentSignal !== signal) {
    if (parentSignal.aborted) onParentAbort();
    else parentSignal.addEventListener("abort", onParentAbort, { once: true });
  }
  const throwIfAborted = () => {
    if (!signal?.aborted) return;
    if (typeof signal.throwIfAborted === "function") signal.throwIfAborted();
    throw new DOMException("操作已取消", "AbortError");
  };
  const runner = async () => {
    while (!stopped && cursor < values.length) {
      try {
        throwIfAborted();
        const index = cursor++;
        results[index] = await worker(values[index], index, { signal });
      } catch (error) {
        stopped = true;
        if (!firstError) firstError = error;
        if (!signal?.aborted) controller?.abort(error);
      }
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(Math.max(1, Math.floor(Number(limit)) || 1), values.length) }, runner));
    if (firstError) throw firstError;
    throwIfAborted();
  } finally {
    if (parentSignal && parentSignal !== signal) parentSignal.removeEventListener("abort", onParentAbort);
  }
  return results;
}

export function resolvedCollageEntries(entries, images) {
  return (Array.isArray(entries) ? entries : []).map((entry, index) => {
    const image = images?.[index];
    const width = Math.max(1, Math.round(finite(image?.naturalWidth ?? image?.width ?? entry?.asset?.naturalWidth ?? entry?.asset?.width, 1)));
    const height = Math.max(1, Math.round(finite(image?.naturalHeight ?? image?.height ?? entry?.asset?.naturalHeight ?? entry?.asset?.height, 1)));
    return { ...entry, asset: { ...entry?.asset, width, height, naturalWidth: width, naturalHeight: height } };
  });
}

export function viewerClipboardPasteIntent({ copied, marker = "", hasImages = false, now = Date.now(), maximumAge = 300_000 } = {}) {
  const validMarker = Boolean(copied?.layer?.id && marker === `XIRAI_LAYER:${copied.layer.id}` && now - Number(copied.copiedAt) >= 0 && now - Number(copied.copiedAt) <= maximumAge);
  if (validMarker) return "internal-layer";
  if (hasImages) return "images";
  return "none";
}

export function normalizeViewerColor(value, fallback = VIEWER_DEFAULT_COLOR) {
  const candidate = String(value || "").trim();
  if (/^#[\da-f]{6}$/i.test(candidate)) return candidate.toLowerCase();
  if (/^#[\da-f]{3}$/i.test(candidate)) return `#${candidate.slice(1).split("").map((part) => part + part).join("")}`.toLowerCase();
  return /^#[\da-f]{6}$/i.test(fallback) ? fallback.toLowerCase() : VIEWER_DEFAULT_COLOR;
}

export function normalizeViewerEdgeLine(value = {}) {
  return {
    enabled: value?.enabled === true,
    style: VIEWER_EDGE_STYLES.includes(value?.style) ? value.style : "solid",
    color: normalizeViewerColor(value?.color),
    width: Math.round(clamp(value?.width ?? 2, 1, 50)),
  };
}

export function normalizeRotation(value) {
  const raw = finite(value, 0);
  return ((raw + 180) % 360 + 360) % 360 - 180;
}






export function viewerSafeResizeHandles(layer) {
  // A rotated layer withdraws every handle: the resize maths assumes an upright box.
  return normalizeRotation(layer?.rotation) !== 0 ? [] : [...VIEWER_RESIZE_HANDLES];
}

export function hasLayerPaint(layer) {
  return Array.isArray(layer?.paintStrokes) && layer.paintStrokes.length > 0;
}

export function hasViewerEdits(layers) {
  return Array.isArray(layers) && layers.some((layer) => hasLayerPaint(layer) || normalizeRotation(layer?.rotation) !== 0);
}

export function clientPointToScene(point, canvasRect, pan = {}, zoom = 1) {
  const safeZoom = Math.max(0.000001, finite(zoom, 1));
  return {
    x: (finite(point?.clientX) - finite(canvasRect?.left) - finite(canvasRect?.width) / 2 - finite(pan?.x)) / safeZoom,
    y: (finite(point?.clientY) - finite(canvasRect?.top) - finite(canvasRect?.height) / 2 - finite(pan?.y)) / safeZoom,
  };
}

export function scenePointToLayer(point, layer) {
  const scale = Math.max(0.000001, finite(layer?.scale, 1));
  const radians = -normalizeRotation(layer?.rotation) * Math.PI / 180;
  const dx = finite(point?.x) - finite(layer?.x);
  const dy = finite(point?.y) - finite(layer?.y);
  const localX = (dx * Math.cos(radians) - dy * Math.sin(radians)) / scale;
  const localY = (dx * Math.sin(radians) + dy * Math.cos(radians)) / scale;
  return {
    x: localX + Math.max(1, finite(layer?.naturalWidth, 1)) / 2,
    y: localY + Math.max(1, finite(layer?.naturalHeight, 1)) / 2,
  };
}

export function clientPointToLayer(point, canvasRect, pan, zoom, layer) {
  return scenePointToLayer(clientPointToScene(point, canvasRect, pan, zoom), layer);
}

export function normalizeStrokePoint(point, width, height) {
  if (!Number.isFinite(Number(point?.x)) || !Number.isFinite(Number(point?.y))) return null;
  const x = Number(point.x);
  const y = Number(point.y);
  const maximumX = Math.max(1, finite(width, 1));
  const maximumY = Math.max(1, finite(height, 1));
  return x >= 0 && x <= maximumX && y >= 0 && y <= maximumY ? { x, y } : null;
}

export function clipLineToLayer(first, second, width, height) {
  if (![first?.x, first?.y, second?.x, second?.y].every((value) => Number.isFinite(Number(value)))) return null;
  const maximumX = Math.max(1, finite(width, 1));
  const maximumY = Math.max(1, finite(height, 1));
  const x0 = Number(first.x);
  const y0 = Number(first.y);
  const dx = Number(second.x) - x0;
  const dy = Number(second.y) - y0;
  let entering = 0;
  let leaving = 1;
  for (const [p, q] of [[-dx, x0], [dx, maximumX - x0], [-dy, y0], [dy, maximumY - y0]]) {
    if (p === 0) {
      if (q < 0) return null;
      continue;
    }
    const ratio = q / p;
    if (p < 0) entering = Math.max(entering, ratio);
    else leaving = Math.min(leaving, ratio);
    if (entering > leaving) return null;
  }
  return [
    { x: x0 + dx * entering, y: y0 + dy * entering },
    { x: x0 + dx * leaving, y: y0 + dy * leaving },
  ];
}

export function clipStrokeSamples(samples, dimensions = {}, previousRawPoint = null) {
  const width = Math.max(1, finite(dimensions.width ?? dimensions.naturalWidth, 1));
  const height = Math.max(1, finite(dimensions.height ?? dimensions.naturalHeight, 1));
  const points = [];
  let previous = previousRawPoint && Number.isFinite(Number(previousRawPoint.x)) && Number.isFinite(Number(previousRawPoint.y))
    ? { x: Number(previousRawPoint.x), y: Number(previousRawPoint.y) }
    : null;
  const append = (point) => {
    const normalized = normalizeStrokePoint(point, width, height);
    const last = points.at(-1);
    if (normalized && (!last || Math.abs(last.x - normalized.x) > 0.01 || Math.abs(last.y - normalized.y) > 0.01)) points.push(normalized);
  };
  for (const sample of Array.isArray(samples) ? samples : []) {
    if (!Number.isFinite(Number(sample?.x)) || !Number.isFinite(Number(sample?.y))) continue;
    const current = { x: Number(sample.x), y: Number(sample.y) };
    if (!previous) append(current);
    else {
      const clipped = clipLineToLayer(previous, current, width, height);
      if (clipped) {
        append(clipped[0]);
        append(clipped[1]);
      }
    }
    previous = current;
  }
  return { points, previousRawPoint: previous };
}

export function normalizePaintStroke(value, dimensions = {}) {
  const tool = value?.tool === "eraser" ? "eraser" : "brush";
  const width = Math.max(1, finite(dimensions.width ?? dimensions.naturalWidth, 1));
  const height = Math.max(1, finite(dimensions.height ?? dimensions.naturalHeight, 1));
  const points = [];
  for (const rawPoint of Array.isArray(value?.points) ? value.points : []) {
    if (points.length >= VIEWER_MAX_STROKE_POINTS) break;
    const point = normalizeStrokePoint(rawPoint, width, height);
    if (!point) continue;
    const previous = points.at(-1);
    if (!previous || Math.abs(previous.x - point.x) > 0.01 || Math.abs(previous.y - point.y) > 0.01) points.push(point);
  }
  const requestedOpacity = finite(value?.opacity ?? 1, 1);
  return {
    id: typeof value?.id === "string" && value.id ? value.id.slice(0, 96) : "",
    tool,
    color: normalizeViewerColor(value?.color),
    size: clamp(value?.size ?? 40, 1, 300),
    opacity: clamp(requestedOpacity > 1 ? requestedOpacity / 100 : requestedOpacity, 0.01, 1),
    points,
  };
}

export function appendStrokePoints(stroke, nextPoints, dimensions) {
  const normalized = normalizePaintStroke({ ...stroke, points: [...(stroke?.points || []), ...(nextPoints || [])] }, dimensions);
  return normalized.points.length === stroke?.points?.length ? stroke : normalized;
}

export function isProtectedViewerHistoryUrl(value) {
  return Boolean(historyAssetIdFromUrl(value));
}





export function viewerEditorLayerBounds(layer) {
  const width = Math.max(1, finite(layer?.naturalWidth, 1)) * clamp(layer?.scale ?? 1, 0.1, 8);
  const height = Math.max(1, finite(layer?.naturalHeight, 1)) * clamp(layer?.scale ?? 1, 0.1, 8);
  const radians = normalizeRotation(layer?.rotation) * Math.PI / 180;
  const aabbWidth = Math.abs(width * Math.cos(radians)) + Math.abs(height * Math.sin(radians));
  const aabbHeight = Math.abs(width * Math.sin(radians)) + Math.abs(height * Math.cos(radians));
  const x = finite(layer?.x);
  const y = finite(layer?.y);
  return { left: x - aabbWidth / 2, right: x + aabbWidth / 2, top: y - aabbHeight / 2, bottom: y + aabbHeight / 2, width: aabbWidth, height: aabbHeight };
}

export function cloneViewerLayer(layer) {
  return {
    ...layer,
    paintStrokes: Array.isArray(layer?.paintStrokes)
      ? layer.paintStrokes.map((stroke) => ({ ...stroke, points: Array.isArray(stroke.points) ? stroke.points.map((point) => ({ ...point })) : [] }))
      : [],
    manualLayout: layer?.manualLayout ? structuredCloneSafe(layer.manualLayout) : layer?.manualLayout,
  };
}

function structuredCloneSafe(value) {
  try {
    if (typeof structuredClone === "function") return structuredClone(value);
  } catch {}
  try { return JSON.parse(JSON.stringify(value)); } catch { return null; }
}

export function cloneViewerLayers(layers) {
  return Array.isArray(layers) ? layers.map(cloneViewerLayer) : [];
}

export function serializeViewerLayer(layer) {
  const paintStrokes = (Array.isArray(layer?.paintStrokes) ? layer.paintStrokes : [])
    .map((stroke) => normalizePaintStroke(stroke, layer)).filter((stroke) => stroke.points.length);
  return Object.fromEntries(Object.entries({
    kind: "image", type: "image", assetId: layer?.assetId || "", url: layer?.url || "",
    originalUrl: layer?.originalUrl || layer?.url || "", name: String(layer?.name || "图片").slice(0, 200),
    naturalWidth: Math.max(1, Math.round(finite(layer?.naturalWidth, 1))), naturalHeight: Math.max(1, Math.round(finite(layer?.naturalHeight, 1))),
    x: finite(layer?.x), y: finite(layer?.y), scale: clamp(layer?.scale ?? 1, 0.1, 8), rotation: normalizeRotation(layer?.rotation),
    mimeType: typeof layer?.mimeType === "string" ? layer.mimeType : "", paintStrokes,
  }).filter(([, value]) => value !== undefined));
}

export function normalizeManualLayout(layout, { trustedCurrentSession = false } = {}) {
  if (!layout || (layout.version !== 1 && layout.version !== 2) || !Array.isArray(layout.layers) || !layout.layers.length || layout.layers.length > 100) return null;
  const version = layout.version;
  let pointCount = 0;
  let strokeCount = 0;
  const layers = [];
  for (const source of layout.layers) {
    if (!source || typeof source !== "object" || Array.isArray(source)) return null;
    // Text layers were withdrawn. A layout saved while they existed still names them, so they are
    // dropped and the pictures around them are kept -- the alternative is refusing to reopen an
    // image that is mostly still readable.
    if (source.kind === "text" || source.type === "text") continue;
    const url = typeof source.originalUrl === "string" && source.originalUrl || typeof source.url === "string" && source.url || "";
    if (!url || (!trustedCurrentSession && !viewerHistorySourcesBound(source))) return null;
    const rawStrokes = source.paintStrokes;
    if (rawStrokes !== undefined && !Array.isArray(rawStrokes)) return null;
    if ((rawStrokes?.length || 0) > VIEWER_MAX_LAYOUT_STROKES - strokeCount) return null;
    for (const stroke of rawStrokes || []) {
      if (!stroke || typeof stroke !== "object" || !Array.isArray(stroke.points) || stroke.points.length > VIEWER_MAX_STROKE_POINTS || pointCount + stroke.points.length > VIEWER_MAX_LAYOUT_POINTS) return null;
      pointCount += stroke.points.length;
      strokeCount += 1;
    }
    const serialized = serializeViewerLayer({ ...source, kind: "image", originalUrl: url, url: typeof source.url === "string" && source.url || url });
    layers.push(serialized);
  }
  const normalized = layers.length ? { version, layers } : null;
  if (!normalized) return null;
  try {
    if (!trustedCurrentSession && new TextEncoder().encode(JSON.stringify(normalized)).byteLength > VIEWER_MAX_LAYOUT_BYTES) return null;
  } catch { return null; }
  return normalized;
}

export function persistedManualLayout(layout, { maxBytes = VIEWER_MAX_LAYOUT_BYTES, maxPoints = VIEWER_MAX_LAYOUT_POINTS } = {}) {
  const normalized = normalizeManualLayout(layout, { trustedCurrentSession: true });
  if (!normalized) return { layout: null, reason: "布局为空" };
  const imageLayers = normalized.layers;
  const protectedSources = imageLayers.every(viewerHistorySourcesBound);
  if (!protectedSources) return { layout: null, reason: "包含本地或非历史图片源" };
  const pointCount = imageLayers.reduce((total, layer) => total + layer.paintStrokes.reduce((sum, stroke) => sum + stroke.points.length, 0), 0);
  if (pointCount > maxPoints) return { layout: null, reason: `笔画点数超过 ${maxPoints}` };
  const bytes = new TextEncoder().encode(JSON.stringify(normalized)).byteLength;
  if (bytes > maxBytes) return { layout: null, reason: `布局数据超过 ${Math.round(maxBytes / 1024)} KiB` };
  return { layout: normalized, reason: "", bytes, pointCount };
}

export function drawPaintStroke(context, rawStroke, dimensions = {}) {
  const stroke = normalizePaintStroke(rawStroke, dimensions);
  if (!stroke.points.length) return;
  const radius = stroke.size / 2;
  context.save();
  context.globalCompositeOperation = stroke.tool === "eraser" ? "destination-out" : "source-over";
  context.globalAlpha = stroke.opacity;
  context.strokeStyle = stroke.tool === "eraser" ? "#000000" : stroke.color;
  context.fillStyle = context.strokeStyle;
  context.lineWidth = stroke.size;
  context.lineCap = "round";
  context.lineJoin = "round";
  if (stroke.points.length === 1) {
    context.beginPath();
    context.arc(stroke.points[0].x, stroke.points[0].y, radius, 0, Math.PI * 2);
    context.fill();
  } else {
    context.beginPath();
    context.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (let index = 1; index < stroke.points.length - 1; index += 1) {
      const point = stroke.points[index];
      const next = stroke.points[index + 1];
      context.quadraticCurveTo(point.x, point.y, (point.x + next.x) / 2, (point.y + next.y) / 2);
    }
    const last = stroke.points.at(-1);
    context.lineTo(last.x, last.y);
    context.stroke();
  }
  context.restore();
}

export function renderRasterLayer(canvas, image, layer) {
  const dimensions = viewerCanvasDimensions(
    Math.max(1, Math.round(finite(layer?.naturalWidth ?? image?.naturalWidth ?? image?.width, 1))),
    Math.max(1, Math.round(finite(layer?.naturalHeight ?? image?.naturalHeight ?? image?.height, 1))),
    "图片编辑画布",
  );
  const { width, height } = dimensions;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: true });
  context.clearRect(0, 0, width, height);
  context.globalCompositeOperation = "source-over";
  context.globalAlpha = 1;
  context.drawImage(image, 0, 0, width, height);
  for (const stroke of Array.isArray(layer?.paintStrokes) ? layer.paintStrokes : []) drawPaintStroke(context, stroke, { width, height });
  return canvas;
}



export function drawViewerLayer(context, layer, source, bounds, outputScale = 1) {
  const scale = clamp(layer?.scale ?? 1, 0.1, 8) * outputScale;
  context.save();
  context.translate((finite(layer?.x) - finite(bounds?.left)) * outputScale, (finite(layer?.y) - finite(bounds?.top)) * outputScale);
  context.rotate(normalizeRotation(layer?.rotation) * Math.PI / 180);
  context.scale(scale, scale);
  context.translate(-Math.max(1, finite(layer?.naturalWidth, 1)) / 2, -Math.max(1, finite(layer?.naturalHeight, 1)) / 2);
  context.drawImage(source, 0, 0, Math.max(1, finite(layer?.naturalWidth, 1)), Math.max(1, finite(layer?.naturalHeight, 1)));
  context.restore();
}
