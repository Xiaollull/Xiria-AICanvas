// QPlan2/P1 Phase B client. The browser submits once and then only reads server snapshots:
// it never loops over cells, draws seeds or explains an old experiment with the current draft.
import { currentLoraRevision } from "./lora-library-client.js";
import { xyzWireDraft } from "./xyz-prompt-source.js";

const ACTIVE = new Set(["queued", "running", "pausing", "paused", "cancelling"]);
export const XYZ_CELL_STATUS_TEXT = {
  pending: "等待", running: "生成中", complete: "已完成", error: "失败", skipped: "已跳过", cancelled: "已取消", missing: "原图缺失",
};
export const XYZ_EXPERIMENT_STATUS_TEXT = {
  queued: "排队中", running: "生成中", pausing: "正在暂停", paused: "已暂停", cancelling: "正在停止",
  complete: "已完成", cancelled: "已停止", interrupted: "已中断", error: "编排错误",
};
export const xyzExperimentActive = (experiment) => Boolean(experiment) && ACTIVE.has(experiment.status);

/** UUID v4 from getRandomValues: `crypto.randomUUID` is absent on plain-HTTP remote hosts. */
export function xyzSubmissionId(random = globalThis.crypto) {
  const bytes = new Uint8Array(16);
  random.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** A transport retry reuses its id only while both the draft and execution policy are unchanged. */
export function xyzSubmissionKey(draftKey, onError) {
  return JSON.stringify([draftKey, onError]);
}

/** Readable text for FastAPI detail shapes: string, {message, errors}, or validation lists. */
export function xyzErrorText(payload, fallback = "请求失败") {
  const detail = payload?.detail ?? payload;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item?.msg || String(item)).join("；") || fallback;
  if (detail && typeof detail === "object") return detail.message || fallback;
  return fallback;
}

export function xyzIssues(payload) {
  const detail = payload?.detail ?? payload;
  return Array.isArray(detail?.errors) ? detail.errors : [];
}

export function xyzCoordinateText(issue) {
  const c = issue?.coordinate;
  if (!c) return issue?.axis ? `${String(issue.axis).toUpperCase()} 轴` : "整体";
  return `X${c.x_index + 1} · Y${c.y_index + 1} · Z${c.z_index + 1} · 样本 ${c.sample_index + 1}`;
}

async function request(fetcher, path, { method = "GET", body, lora = false, signal } = {}) {
  const headers = { "Content-Type": "application/json" };
  // The same published revision the ordinary job route stamps; Node and Python both compare it.
  if (lora) headers["X-XirAI-LoRA-Revision"] = String(currentLoraRevision());
  const response = await fetcher(path, { method, headers, cache: "no-store", signal, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw Object.assign(new Error(xyzErrorText(payload, `请求失败（${response.status}）`)), { status: response.status, payload });
  return payload;
}

export function xyzApi(fetcher = globalThis.fetch.bind(globalThis)) {
  const base = "/api/inference/xyz-plots";
  return {
    capabilities: (options) => request(fetcher, `${base}/capabilities`, options),
    plan: (draft, options) => request(fetcher, `${base}/plan`, { ...options, method: "POST", body: xyzWireDraft(draft), lora: true }),
    submit: (draft, { submissionId, onError = "pause", ...options } = {}) => request(fetcher, base, { ...options, method: "POST", lora: true,
      body: { schema_version: 1, kind: "xyz_submit", submission_id: submissionId, draft: xyzWireDraft(draft), on_error: onError } }),
    // origin=none: only the XYZ workspace's own experiments; LoRA-mix rounds (an `origin` block) belong to the search.
    list: (engine, options) => request(fetcher, `${base}?engine=${encodeURIComponent(engine)}&origin=none`, options),
    tasks: (engine, options) => request(fetcher, `${base}/tasks${engine ? `?engine=${encodeURIComponent(engine)}` : ""}`, options),
    previewTaskDeletion: (body, options) => request(fetcher, `${base}/task-deletions/preview`, { ...options, method: "POST", body }),
    deleteTasks: (body, options) => request(fetcher, `${base}/task-deletions`, { ...options, method: "POST", body }),
    detail: (id, options) => request(fetcher, `${base}/${encodeURIComponent(id)}`, options),
    cell: (id, cellId, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/cells/${encodeURIComponent(cellId)}`, options),
    control: (id, action, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/${action}`, { ...options, method: "POST" }),
    decide: (id, action, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/decision`, { ...options, method: "POST", body: { action } }),
    estimateExport: (id, body, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/export/estimate`, { ...options, method: "POST", body }),
    createExport: (id, body, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/exports`, { ...options, method: "POST", body }),
    exportStatus: (id, exportId, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/exports/${encodeURIComponent(exportId)}`, options),
    cancelExport: (id, exportId, options) => request(fetcher, `${base}/${encodeURIComponent(id)}/exports/${encodeURIComponent(exportId)}/cancel`, { ...options, method: "POST" }),
    continue: (id, retry = "none", { requestedConcurrency, ...options } = {}) => request(fetcher, `${base}/${encodeURIComponent(id)}/continue`, { ...options, method: "POST",
      body: { retry, ...(Number.isInteger(requestedConcurrency) ? { requested_concurrency: requestedConcurrency } : {}) } }),
  };
}

/** Rows (Y) × columns (X) of one Z/sample page, addressed by stable coordinates. */
export function xyzGridPage(detail, zIndex, sampleIndex) {
  const columns = detail.axes.x.labels.length || 1, rows = detail.axes.y.labels.length || 1;
  const grid = Array.from({ length: rows }, () => Array.from({ length: columns }, () => null));
  for (const cell of detail.cells) {
    const [x, y, z, sample] = cell.coordinate;
    if (z === zIndex && sample === sampleIndex) grid[y][x] = cell;
  }
  return grid;
}

export const XYZ_EXPORT_STATUS_TEXT = { queued: "排队中", running: "导出中", complete: "已完成", cancelled: "已取消", error: "失败", interrupted: "已中断" };
export const xyzExportActive = (record) => Boolean(record) && ["queued", "running"].includes(record.status);

/** The request body for one set of dialog choices. Original is always 100 %; a page scope carries the page. */
export function xyzExportBody(options, page) {
  return {
    scope: options.scope, page: options.scope === "page" ? { z_index: page?.z ?? 0, sample_index: page?.sample ?? 0 } : null,
    mode: options.mode, percent: options.mode === "original" ? 100 : options.percent,
    theme: options.theme, arrangement: options.arrangement, captions: options.captions,
  };
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes, unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}
