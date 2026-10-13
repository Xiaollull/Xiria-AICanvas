import { createRelinkDraft, editRelinkRow, selectedRelinkRows, restoreRelinkSuggestion } from "./lora-relink.js";
import { loraFetch, announceLoraPublication } from "./lora-library-client.js";

export const initialRepairState = () => ({ phase: "idle", scan: null, draft: null, preview: null, error: "", receipt: null, available: false, registerBaseline: false,
  params: null, scannedParams: null, staleParams: false, directories: null, scopeCheck: null });
// P10.2 scan parameters: engines + problematic source folder + target folder.
const paramsKey = (params) => JSON.stringify({ engines: params?.engines || [], source: params?.source || "", target: params?.target || "" });
const scanBody = (params) => ({ engines: params.engines, ...(params.source ? { source: params.source } : {}), ...(params.target ? { target: params.target } : {}) });
/** One controller for both entry points. Draft edits invalidate preview/idempotent
 * payload, never apply associations. Late results are epoch/selection gated.
 */
export function createLoraRepairController({ request = loraFetch, changed = () => {} } = {}) {
  let state = initialRepairState(), epoch = 0, scopeEpoch = 0, abort, pendingCommit;
  const publish = (patch) => { state = { ...state, ...patch }; changed(state); return state; };
  const call = async (action, payload, signal) => {
    const response = await request(`/api/lora-relink/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
    const value = await response.json();
    if (!response.ok) throw Object.assign(new Error(value.error || "修复请求失败；草稿已保留。"), { status: response.status, conflicts: value.conflicts });
    return value;
  };
  const payload = () => ({ scanId: state.scan.scanId, selections: state.draft.rows.map((row) => ({ sourceId: row.sourceId, target: row.target,
    scope: row.scope, skipped: row.skipped, confirmedOrigin: row.target?.kind === "relocate" })), registerBaseline: state.registerBaseline });
  return {
    state: () => state,
    async availability() {
      try { const response = await request("/api/lora-relink/state", { cache: "no-store" }); const value = await response.json(); publish({ available: response.ok && value.available && value.contractVersion === 1 }); }
      catch { publish({ available: false }); }
    },
    /** Engines array (back-compat) or { engines, source, target }. */
    async scan(input, { discard = false } = {}) {
      if (state.draft && selectedRelinkRows(state.draft).length && !discard) return false;
      const params = Array.isArray(input) ? { engines: input, source: "", target: "" } : input;
      const captured = ++epoch; abort?.abort(); abort = new AbortController();
      publish({ phase: "scanning", error: "", preview: null, params });
      try {
        const scan = await call("scan", scanBody(params), abort.signal);
        if (captured !== epoch) return false;
        const draft = createRelinkDraft(scan, scan.rows, scan.revision);
        // A new scan is a new draft: a failed commit's idempotent payload
        // (old scanId/operationId) must not be retried against it.
        pendingCommit = null;
        publish({ phase: "editing", scan, draft, receipt: null, registerBaseline: false, scannedParams: paramsKey(params), staleParams: false }); return true;
      } catch (error) { if (captured === epoch) publish({ phase: state.draft ? "editing" : "idle", error: error.name === "AbortError" ? "扫描已取消，未应用任何关联。" : error.message }); return false; }
    },
    /** Changing engine/source/target invalidates preview, pending payload and confirmation until a rescan. */
    setParams(params) {
      ++epoch; pendingCommit = null;
      publish({ params, preview: null, staleParams: Boolean(state.scannedParams) && paramsKey(params) !== state.scannedParams });
    },
    /** Folder navigation + inline server validation; no hashing. Late answers are dropped. */
    async loadScope(params) {
      const captured = ++scopeEpoch;
      try {
        const value = await call("scope", scanBody(params));
        if (captured === scopeEpoch) publish({ directories: value.directories, scopeCheck: { source: value.source || null, target: value.target || null } });
      } catch (error) { if (captured === scopeEpoch) publish({ scopeCheck: { error: error.message } }); }
    },
    /** Explicit confirmed-deletion marking; applies nothing until preview + confirm. */
    markDeleted(sourceIds) {
      ++epoch; pendingCommit = null;
      let draft = state.draft;
      // An unused entry listed under processed becomes an explicit deletion row.
      const moving = (draft.processed || []).filter((row) => sourceIds.includes(row.sourceId));
      if (moving.length) draft = { ...draft, processed: draft.processed.filter((row) => !sourceIds.includes(row.sourceId)),
        rows: [...draft.rows, ...moving.map((row) => ({ ...row, target: null, chosenBy: "suggestion", skipped: false, scope: { all: true, history: true }, needsReview: false }))] };
      for (const sourceId of sourceIds) draft = editRelinkRow(draft, sourceId, { target: { kind: "delete" }, targetInput: undefined, skipped: false, scope: { all: true } });
      publish({ draft, preview: null, error: "", phase: "editing" });
    },
    cancel() { ++epoch; abort?.abort(); publish({ phase: state.draft ? "editing" : "idle", error: "扫描已取消，未应用任何关联。" }); },
    edit(sourceId, patch) { ++epoch; pendingCommit = null; publish({ draft: editRelinkRow(state.draft, sourceId, patch), preview: null, error: "", phase: "editing" }); },
    restore(sourceId) { ++epoch; pendingCommit = null; publish({ draft: restoreRelinkSuggestion(state.draft, sourceId), preview: null, error: "", phase: "editing" }); },
    /** Explicit batch skip/unskip; targets and scopes are kept. */
    skipRows(sourceIds, skipped) {
      ++epoch; pendingCommit = null;
      const ids = new Set(sourceIds);
      publish({ draft: { ...state.draft, rows: state.draft.rows.map((row) => ids.has(row.sourceId) ? { ...row, skipped, chosenBy: "user" } : row) }, preview: null, error: "", phase: "editing" });
    },
    baseline(enabled) { ++epoch; pendingCommit = null; publish({ registerBaseline: enabled, preview: null }); },
    markRevision(next) {
      // The transport observes our receipt's published revision before commit()
      // consumes its body. Do not label our successful result as a stale draft.
      if (["committing", "undoing"].includes(state.phase) || next === state.receipt?.revision) return;
      if (state.draft && next !== state.draft.revision) publish({ draft: { ...state.draft, rows: state.draft.rows.map((row) => ({ ...row, needsReview: true })) }, preview: null, error: "发布版本已改变；选择已保留，请重新扫描并复核。" });
    },
    async preview() {
      if (state.staleParams) return;
      const captured = epoch; publish({ phase: "previewing", error: "" });
      try { const preview = await call("preview", payload()); if (captured === epoch) publish({ preview, phase: "editing" }); }
      catch (error) { if (captured === epoch) publish({ phase: "editing", error: error.message, preview: error.conflicts ? { conflicts: error.conflicts } : null }); }
    },
    async commit() {
      if (!state.available || !state.scan?.complete || state.staleParams || !state.preview || state.preview.conflicts?.length || state.draft.rows.some((row) => !row.skipped && row.needsReview)) return;
      pendingCommit ||= { ...payload(), operationId: crypto.randomUUID() };
      publish({ phase: "committing", error: "" });
      try { const receipt = await call("commit", pendingCommit); publish({ receipt: { ...receipt, affected: state.preview.affected }, phase: "result", error: "" }); announceLoraPublication(receipt.revision); }
      catch (error) { publish({ phase: "editing", error: `${error.message} 可保留选择后复核或用相同操作重试。`, preview: error.conflicts ? { conflicts: error.conflicts } : state.preview }); }
    },
    async undo() {
      if (!state.receipt?.operationId) return;
      publish({ phase: "undoing", error: "" });
      try { const receipt = await call("undo", { operationId: state.receipt.operationId, undoOperationId: crypto.randomUUID() }); publish({ receipt: { ...receipt, undone: true }, phase: "result" }); announceLoraPublication(receipt.revision); }
      catch (error) { publish({ phase: "result", error: error.message }); }
    },
  };
}
