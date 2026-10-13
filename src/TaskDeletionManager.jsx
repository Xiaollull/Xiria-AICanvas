import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Trash2, X } from "lucide-react";
import { useDialogLifecycle } from "./gallery-core.js";
import { XYZ_EXPERIMENT_STATUS_TEXT } from "./xyz-client.js";
import { TASK_SELECTION_LIMIT, deletionBody, deletionResult } from "./task-deletion.js";

const when = (task) => task.created_at ? new Date(task.created_at * 1000).toLocaleString() : task.id.slice(0, 8);
const statusText = (task) => XYZ_EXPERIMENT_STATUS_TEXT[task.status] || task.status || "—";
/** The server's retain reason is exact but long; the row shows its gist and keeps the full text as the tooltip. */
const reasonGist = (reason) => {
  const text = String(reason || "");
  if (/运行 \/ 排队 \/ 暂停中/.test(text)) return `${text.match(/^第 \d+ 轮：/)?.[0] || ""}运行、排队或暂停中`;
  if (text.includes("导出")) return "正在导出";
  if (text.includes("其他存活进程")) return "其他进程拥有";
  if (text.includes("队列") || text.includes("执行器")) return "队列 / 执行器仍在使用";
  return text;
};

/** The task list and the deletion preview share ONE dialog: Escape on the preview returns to the list, on the list closes. */
function ManagerDialog({ module, label, engine, listing, checked, setChecked, preview, result, error, busy, currentId, onRefresh, onPrepare, onCancel, onConfirm, onClose }) {
  const ref = useDialogLifecycle(true, () => { if (busy) return; if (preview) onCancel(); else onClose(); });
  const tasks = listing?.tasks || [], eligible = tasks.filter((task) => task.deletable);
  const failedIds = result?.results.filter((item) => item.outcome === "failed").map((item) => item.id) || [];
  const mix = module === "lora_mix", total = listing?.module_total;
  const toggle = (task, on) => setChecked((ids) => on ? [...ids, task.id] : ids.filter((id) => id !== task.id));
  return createPortal(<div className="xyz-sync-backdrop"><section ref={ref} className={`xyz-sync-dialog task-delete-manager panel${preview ? " task-delete-dialog" : ""}`} role="dialog" aria-modal="true" aria-labelledby="task-delete-title" aria-busy={busy}>
    {preview ? <>
      <header className="task-delete-head"><div><h2 id="task-delete-title">删除 {label} 任务记录？</h2>
        <span className="task-delete-scope">{preview.scope === "all" ? `范围：仅 ${label} 模块的所有引擎、全部任务（由服务器枚举）` : `范围：仅 ${label} 模块中勾选的任务`} · 共 {preview.total} 个任务，可删除 {preview.eligible} 个，保留 {preview.skipped} 个</span></div></header>
      <div className="task-delete-callout"><strong>保留所有生成原图、已导出网格与画廊副本。</strong>只从任务列表移除记录，不擦除文件；删除后不能继续旧任务。其他模块、草稿、LoRA 组合与提示词不变。
        {mix && <> 每个搜索任务包含全部轮次；任何一轮活动则整任务保留。历史已测试 / 已预留候选仍用于排重，不自动释放预留或改变预算。</>}</div>
      <ul className={`task-delete-preview${mix ? " with-rounds" : ""}`}>{preview.tasks.map((task) => <li key={task.id} className={task.deletable ? "" : "kept"} title={task.reason || task.id}>
        <span className="task-delete-engine">{task.engine || "—"}</span><span className="task-delete-when">{when(task)}</span><code>{task.id.slice(0, 8)}</code>
        {mix && <span>{task.rounds ? `${task.rounds} 轮` : ""}</span>}
        <small>{task.deletable ? "将删除任务记录" : `保留：${task.reason}`}</small></li>)}</ul>
      {preview.unreadable?.length > 0 && <p className="task-delete-note" role="status">另有 {preview.unreadable.length} 条无法安全读取的记录，原文件保留，不作为已删除任务报告。</p>}
      {error && <p className="xyz-error" role="alert">{error}</p>}
      <footer className="task-delete-foot"><span /><button type="button" disabled={busy} onClick={onCancel}>取消</button>
        <button type="button" className="task-delete-danger" disabled={busy || !preview.eligible} onClick={onConfirm}>{busy ? "正在删除…" : `确认删除 ${preview.eligible} 个任务`}</button></footer>
    </> : <>
      <header className="task-delete-head"><div><h2 id="task-delete-title">管理 {mix ? "LoRA 混合搜索任务" : "XYZ 实验任务"}</h2>
        <span className="task-delete-scope">{engine} · {tasks.length} 个{mix ? "搜索（含各轮）" : "实验"}{Number.isInteger(total) && total !== tasks.length ? ` · 本模块全部引擎共 ${total} 个` : ""}</span></div>
        <button type="button" className="task-delete-close" aria-label="关闭任务管理" title="关闭" disabled={busy} onClick={onClose}><X size={16} /></button></header>
      <p className="task-delete-note">删除只移除任务记录；生成原图、已导出网格与画廊副本全部保留。运行、排队或暂停中的任务不可删除。</p>
      {result && <div className="task-delete-receipt" role="status"><p>已删除 {result.deleted} 个 · 保留 {result.skipped} 个 · 失败 {result.failed} 个。图片与导出文件均保留。</p>
        {failedIds.length > 0 && <button type="button" disabled={busy} onClick={() => onPrepare("selected", failedIds.slice(0, TASK_SELECTION_LIMIT))}>重试失败任务…</button>}
        <details><summary>查看逐项结果</summary><ul>{result.results.map((item) => <li key={item.id}>{item.id.slice(0, 8)} · {{ deleted: "已删除任务记录", skipped: "保留", failed: "失败" }[item.outcome]}{item.reason ? `：${item.reason}` : ""}</li>)}</ul></details></div>}
      {error && <p className="xyz-error" role="alert">{error}</p>}
      <div className="task-delete-toolbar">
        <span>已选 <b>{checked.length}</b> / 可删除 {eligible.length}{eligible.length > TASK_SELECTION_LIMIT ? "（单次最多 200；“删除全部”不受此限）" : ""}</span>
        <button type="button" disabled={busy || !eligible.length || checked.length === Math.min(eligible.length, TASK_SELECTION_LIMIT)} onClick={() => setChecked(eligible.slice(0, TASK_SELECTION_LIMIT).map((task) => task.id))}>全选</button>
        <button type="button" disabled={busy || !checked.length} onClick={() => setChecked([])}>取消选择</button>
        <button type="button" disabled={busy} onClick={onRefresh}>刷新</button>
      </div>
      <div className={`task-delete-list${mix ? " with-rounds" : ""}`} role="group" aria-label={`${engine} 的 ${label} 任务`}>
        {tasks.map((task) => {
          const on = checked.includes(task.id), locked = busy || !task.deletable || (!on && checked.length >= TASK_SELECTION_LIMIT);
          return <label key={task.id} className={`${task.deletable ? "" : "kept"}${on ? " checked" : ""}`} title={task.reason || task.id}>
            <input type="checkbox" aria-label={`勾选任务 ${task.id.slice(0, 8)}`} disabled={locked} checked={on} onChange={(event) => toggle(task, event.target.checked)} />
            <span className="task-delete-when">{when(task)}</span>
            <span className={`task-delete-status s-${task.status}`}>{statusText(task)}</span>
            {mix && <span className="task-delete-rounds">{task.rounds} 轮</span>}
            <small>{task.deletable ? (task.id === currentId || task.experiment_ids?.includes(currentId) ? "可删除 · 当前查看" : "可删除") : `保留：${reasonGist(task.reason)}`}</small>
          </label>;
        })}
        {!tasks.length && <p className="task-delete-empty" role="status">{listing ? `${engine} 没有 ${label} 任务；仍可用“删除全部”清理本模块其他引擎的任务。` : "正在读取任务清单…"}</p>}
      </div>
      {listing?.unreadable?.length > 0 && <p className="task-delete-note">{listing.unreadable.length} 条记录无法安全读取，原文件保留。</p>}
      <footer className="task-delete-foot">
        <button type="button" className="task-delete-danger-ghost" disabled={busy || !listing || !total} title="由服务器枚举本模块全部引擎的任务，确认前会列出将删除与保留的项目" onClick={() => onPrepare("all")}>删除全部任务（全部引擎{Number.isInteger(total) ? ` · ${total} 个` : ""}）…</button>
        <span />
        <button type="button" disabled={busy} onClick={onClose}>关闭</button>
        <button type="button" className="task-delete-danger" disabled={busy || !checked.length} onClick={() => onPrepare("selected")}><Trash2 size={13} aria-hidden="true" />删除选中{checked.length ? ` ${checked.length} 个` : ""}…</button>
      </footer>
    </>}
  </section></div>, document.body);
}

/**
 * Task deletion for one result module. Renders a compact entry button for the results header; the list, selection,
 * preview and receipt live in a dialog. All request/epoch guards stay in this component, which remains mounted while
 * the dialog opens and closes, so a late response can never act on another engine's or another selection's view.
 */
export default function TaskDeletionManager({ api, module, engine, refreshToken, onDeleted, currentId = null }) {
  const label = module === "xyz" ? "XYZ" : "LoRA 混合搜索";
  const [listing, setListing] = useState(null), [checked, setChecked] = useState([]), [preview, setPreview] = useState(null), [result, setResult] = useState(null), [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const context = `${module}:${engine}`, liveContext = useRef(context), alive = useRef(true), epoch = useRef(0), listEpoch = useRef(0), flight = useRef(false), callback = useRef(onDeleted);
  liveContext.current = context; callback.current = onDeleted;
  const load = useCallback(async () => {
    const token = ++listEpoch.current, key = context;
    try {
      const value = await api.tasks(engine);
      if (!alive.current || key !== liveContext.current || token !== listEpoch.current) return;
      if (!Array.isArray(value?.tasks)) throw new Error("任务清单响应无效；请刷新重试");
      setListing(value); setChecked((ids) => ids.filter((id) => value.tasks.some((task) => task.id === id && task.deletable)));
    } catch (failure) { if (alive.current && key === liveContext.current && token === listEpoch.current) setError(failure.message); }
  }, [api, engine, context]);
  const reload = useRef(load); reload.current = load;
  useEffect(() => { alive.current = true; return () => { alive.current = false; epoch.current++; listEpoch.current++; }; }, []);
  useEffect(() => { epoch.current++; listEpoch.current++; setListing(null); setChecked([]); setPreview(null); setResult(null); setError(""); setBusy(flight.current); void load(); return () => { epoch.current++; listEpoch.current++; }; }, [load]);
  useEffect(() => { void load(); }, [refreshToken, load]);
  const prepare = async (scope, ids = checked) => {
    if (flight.current) return;
    const body = deletionBody(scope, ids), token = ++epoch.current, key = context;
    flight.current = true; setBusy(true); setError("");
    try {
      const value = await api.previewTaskDeletion(body);
      if (!alive.current || key !== liveContext.current || token !== epoch.current) return;
      if (value?.module !== module || value.scope !== scope || !Array.isArray(value.tasks) || !Number.isInteger(value.eligible) || typeof value.fingerprint !== "string") throw new Error("无法安全核对删除范围，请刷新重试");
      setPreview({ ...value, body });
    } catch (failure) { if (alive.current && key === liveContext.current && token === epoch.current) setError(failure.message); }
    finally { flight.current = false; if (alive.current) setBusy(false); }
  };
  const cancel = () => { if (flight.current) return; epoch.current++; setPreview(null); setError(""); };
  const confirm = async () => {
    if (flight.current || !preview?.eligible) return;
    const token = ++epoch.current, key = context, snapshot = preview;
    flight.current = true; setBusy(true); setError(""); setResult(null);
    try {
      const value = deletionResult(await api.deleteTasks({ ...snapshot.body, fingerprint: snapshot.fingerprint }));
      if (!alive.current) return;
      if (value.module !== module) throw new Error("删除凭据的模块不匹配，请刷新核对");
      if (key !== liveContext.current || token !== epoch.current) {
        // A committed ALL may include the new engine too. Reconcile through the NEW owner callback,
        // never revive the old selection/confirmation or reuse its list closure.
        if (liveContext.current.startsWith(`${module}:`)) { await callback.current?.(value); await reload.current(); }
        return;
      }
      listEpoch.current++; setResult(value); setPreview(null);
      setChecked((ids) => ids.filter((id) => !value.results.some((item) => item.id === id && item.outcome === "deleted")));
      await callback.current?.(value); await load();
    } catch (failure) {
      if (alive.current && key === liveContext.current && token === epoch.current) {
        setError(`${failure.message}；删除结果未获确认，请刷新核对后重试`); setPreview(null);
        try {
          const fresh = await api.tasks(engine);
          if (!alive.current || key !== liveContext.current || token !== epoch.current || !Array.isArray(fresh?.tasks)) return;
          const visible = new Set(fresh.tasks.map((task) => task.id));
          const absent = snapshot.tasks.filter((task) => task.deletable && task.engine === engine && !visible.has(task.id));
          // Reconcile authoritative absence, but do not emit a fabricated success receipt.
          await callback.current?.({ results: [], visibilityRemoved: absent.flatMap((task) => task.experiment_ids), sessionVisibilityRemoved: absent.map((task) => task.id) });
          if (alive.current && key === liveContext.current && token === epoch.current) {
            setListing(fresh); setChecked((ids) => ids.filter((id) => visible.has(id)));
          }
        } catch { /* Keep the explicit unknown-result error; user can retry a read later. */ }
      }
    }
    finally { flight.current = false; if (alive.current) setBusy(false); }
  };
  const tasks = listing?.tasks || [];
  const openManager = () => { setOpen(true); setError(""); void load(); };
  const close = () => { if (flight.current) return; epoch.current++; setPreview(null); setOpen(false); };
  return <>
    <button type="button" className="task-delete-entry" aria-haspopup="dialog" aria-expanded={open} title={`勾选删除或一键删除${label}任务记录（原图与导出文件保留）`} onClick={openManager}>
      <Trash2 size={13} aria-hidden="true" />管理任务{listing ? <b>{tasks.length}</b> : null}</button>
    {open && <ManagerDialog module={module} label={label} engine={engine} listing={listing} checked={checked} setChecked={setChecked} preview={preview} result={result} error={error} busy={busy} currentId={currentId}
      onRefresh={() => { setError(""); void load(); }} onPrepare={(scope, ids) => void prepare(scope, ids)} onCancel={cancel} onConfirm={() => void confirm()} onClose={close} />}
  </>;
}
