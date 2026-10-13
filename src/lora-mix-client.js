// QPlan2/P2 client. Like the XYZ client it submits once and afterwards only reads server snapshots: the browser
// never reserves vectors, never decides what is tested, and never explains an old round with the current draft.
import { currentLoraRevision } from "./lora-library-client.js";
import { loraMixRecipe } from "./lora-mix-draft.js";
import { xyzApi, xyzErrorText } from "./xyz-client.js";

async function request(fetcher, path, { method = "GET", body, lora = false, signal } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (lora) headers["X-XirAI-LoRA-Revision"] = String(currentLoraRevision());
  const response = await fetcher(path, { method, headers, cache: "no-store", signal, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw Object.assign(new Error(xyzErrorText(payload, `请求失败（${response.status}）`)), { status: response.status, payload });
  return payload;
}

export function loraMixApi(fetcher = globalThis.fetch.bind(globalThis)) {
  const base = "/api/inference/lora-mix";
  return {
    capabilities: (options) => request(fetcher, `${base}/capabilities`, options),
    used: (draft, options) => request(fetcher, `${base}/used`, { ...options, method: "POST", body: { draft: loraMixRecipe(draft) } }),
    plan: (draft, options) => request(fetcher, `${base}/plan`, { ...options, method: "POST", lora: true, body: { draft: loraMixRecipe(draft) } }),
    confirm: (draft, { submissionId, onError = "pause", ...options } = {}) => request(fetcher, `${base}/sessions`, { ...options, method: "POST", lora: true,
      body: { schema_version: 1, kind: "lora_mix_submit", submission_id: submissionId, draft: loraMixRecipe(draft), on_error: onError } }),
    sessions: (engine, options) => request(fetcher, `${base}/sessions?engine=${encodeURIComponent(engine)}`, options),
    tasks: (engine, options) => request(fetcher, `${base}/tasks${engine ? `?engine=${encodeURIComponent(engine)}` : ""}`, options),
    previewTaskDeletion: (body, options) => request(fetcher, `${base}/task-deletions/preview`, { ...options, method: "POST", body }),
    deleteTasks: (body, options) => request(fetcher, `${base}/task-deletions`, { ...options, method: "POST", body }),
    session: (id, options) => request(fetcher, `${base}/sessions/${encodeURIComponent(id)}`, options),
    remaining: (id, { format = "json", offset = 0, limit = 200, round, ...options } = {}) => request(fetcher, `${base}/sessions/${encodeURIComponent(id)}/remaining?format=${format}&offset=${offset}&limit=${limit}${Number.isInteger(round) ? `&round=${round}` : ""}`, options),
    release: (experimentId, options) => request(fetcher, `${base}/experiments/${encodeURIComponent(experimentId)}/release`, { ...options, method: "POST" }),
    // Candidate-grid export: same renderer and records as the XYZ grid, laid out in rows x columns. Status/cancel are the XYZ routes.
    exports: (() => {
      const xyz = xyzApi(fetcher), root = (id) => `${base}/experiments/${encodeURIComponent(id)}`;
      return {
        estimateExport: (id, body, options) => request(fetcher, `${root(id)}/export/estimate`, { ...options, method: "POST", body }),
        createExport: (id, body, options) => request(fetcher, `${root(id)}/exports`, { ...options, method: "POST", body }),
        exportStatus: xyz.exportStatus, cancelExport: xyz.cancelExport,
      };
    })(),
  };
}
