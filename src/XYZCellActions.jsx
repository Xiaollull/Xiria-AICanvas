// QPlan2/P1 Phase E: a cell's actual settings applied to Basic Generation, chosen group by group, and
// a grid PNG read back to its experiment. Shared by the XYZ results grid and the image reader. Nothing
// here applies anything until the user presses Apply; reading a grid never applies a cell.
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Check, X } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import { useDialogLifecycle } from "./gallery-core.js";
import { XYZ_APPLY_DEFAULT_GROUPS, xyzApplyRows, xyzBasicPrompt, xyzCellSettings } from "./xyz-apply.js";
import { XYZ_CELL_STATUS_TEXT, XYZ_EXPERIMENT_STATUS_TEXT, xyzApi } from "./xyz-client.js";

const api = xyzApi();

function cellTitle(cell) {
  const [x, y, z, sample] = cell?.coordinate && Array.isArray(cell.coordinate)
    ? cell.coordinate
    : [cell?.coordinate?.x_index, cell?.coordinate?.y_index, cell?.coordinate?.z_index, cell?.coordinate?.sample_index];
  const labels = cell?.labels ? Object.values(cell.labels).filter((label) => label && label !== "—").join(" / ") : "";
  return `X${(x ?? 0) + 1} · Y${(y ?? 0) + 1} · Z${(z ?? 0) + 1} · 样本 ${(sample ?? 0) + 1}${labels ? ` · ${labels}` : ""}`;
}

/** Apply chosen groups of one cell's frozen request to Basic. `promptContext` describes Basic now. */
export function XYZCellApplyDialog({ experimentId, cellId, promptContext, onApply, onGoBasic, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const [cell, setCell] = useState(null), [loadError, setLoadError] = useState("");
  const [groups, setGroups] = useState(() => new Set(XYZ_APPLY_DEFAULT_GROUPS));
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [applied, setApplied] = useState(null);
  useEffect(() => {
    let stopped = false;
    api.cell(experimentId, cellId).then((value) => { if (!stopped) setCell(value); }).catch((failure) => { if (!stopped) setLoadError(failure.message); });
    return () => { stopped = true; };
  }, [experimentId, cellId]);
  const applyingLoras = groups.has("loras");
  const prompt = useMemo(() => cell?.prompt_parts ? xyzBasicPrompt(cell.prompt_parts, {
    ...promptContext,
    // Applying LoRAs switches off combinations whose members are no longer mounted, so their words
    // stop leading the prompt; the comparison is drawn for the state the apply leaves behind.
    groupPrompt: promptContext?.groupPromptFor ? promptContext.groupPromptFor(applyingLoras ? cell.lora_references || [] : null) : "",
  }) : null, [cell, promptContext, applyingLoras]);
  const settings = useMemo(() => cell && prompt ? xyzCellSettings(cell, prompt) : null, [cell, prompt]);
  const rows = useMemo(() => settings ? xyzApplyRows(settings) : [], [settings]);
  const toggle = (id) => setGroups((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const run = async (chosen) => {
    setBusy(true); setError("");
    try { await onApply(settings, chosen); setApplied(chosen); }
    catch (failure) { setError(failure.message || "参数应用失败"); }
    finally { setBusy(false); }
  };
  const complete = cell?.status === "complete";
  return createPortal(<div className="xyz-sync-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <section className="xyz-sync-dialog xyz-apply-dialog panel" role="dialog" aria-modal="true" aria-labelledby="xyz-apply-title" ref={ref}>
      <header className="xyz-apply-head"><div><h2 id="xyz-apply-title">把这一格应用到基础生图</h2><p>{cell ? cellTitle(cell) : cellId}</p></div>
        <button type="button" className="xyz-apply-close" aria-label="关闭应用窗口" disabled={busy} onClick={onClose}><X size={16} /></button></header>
      <p>按本格<strong>实际执行</strong>的冻结请求写入；只写所选分组。实验、其他格与真实 LoRA 组合都不改动。默认只选提示词。</p>
      {loadError && <p role="alert">{loadError}</p>}
      {!cell && !loadError && <p role="status">正在读取本格的冻结请求…</p>}
      {cell && !complete && <p role="alert">本格状态为「{XYZ_CELL_STATUS_TEXT[cell.status] || cell.status}」，没有生成过原图；仍可应用它计划使用的参数。</p>}
      {settings && <>
        <ul className="xyz-apply-groups">{rows.map((row) => <li key={row.id}>
          <button type="button" role="checkbox" aria-checked={groups.has(row.id)} disabled={busy} onClick={() => toggle(row.id)}>
            <i>{groups.has(row.id) && <Check size={11} />}</i><span><strong>{row.label}</strong><small title={row.summary}>{row.summary}</small></span>
          </button></li>)}</ul>
        <section className="xyz-apply-prompt" aria-label="提示词对照">
          <label>写入正向提示词框<textarea readOnly rows={3} value={prompt.positive} /></label>
          {cell.request?.negative_prompt !== undefined && <label>写入负向提示词框<textarea readOnly rows={2} value={prompt.negative} /></label>}
          {prompt.notes.map((note) => <p key={note} className="xyz-note">{note}</p>)}
          <p className={prompt.reproduces.positive && prompt.reproduces.negative ? "xyz-apply-match" : "xyz-warning"}>
            {prompt.reproduces.positive && prompt.reproduces.negative ? "✓ 按基础生图当前的组合与预设，生成时的实际提示词与本格完全一致" : "✗ 生成时的实际提示词与本格不同"}</p>
          {!prompt.reproduces.positive && <div className="xyz-apply-compare"><label>本格实际正向<textarea readOnly rows={3} value={cell.prompt_parts.prompt} /></label><label>基础生图将发送<textarea readOnly rows={3} value={prompt.preview.positive} /></label></div>}
          {!prompt.reproduces.negative && <div className="xyz-apply-compare"><label>本格实际负向<textarea readOnly rows={2} value={cell.prompt_parts.negative_prompt} /></label><label>基础生图将发送<textarea readOnly rows={2} value={prompt.preview.negative} /></label></div>}
        </section>
      </>}
      {error && <p role="alert">{error}</p>}
      {applied && <div className="xyz-apply-done" role="status"><p>已把 {applied.length} 组参数写入基础生图（{applied.map((id) => rows.find((row) => row.id === id)?.label || id).join("、")}）。</p>
        {onGoBasic && <button type="button" onClick={onGoBasic}>前往基础生图</button>}</div>}
      <div className="xyz-actions">
        <button type="button" disabled={busy} onClick={onClose}>{applied ? "关闭" : "取消"}</button>
        <button type="button" disabled={busy || !settings} onClick={() => void run(rows.map((row) => row.id))}>应用全部 {rows.length} 组</button>
        <button type="button" className="primary" disabled={busy || !settings || !groups.size} onClick={() => void run(rows.filter((row) => groups.has(row.id)).map((row) => row.id))}>{busy ? "正在应用…" : `应用所选 ${groups.size} 组`}</button>
      </div>
    </section>
  </div>, document.body);
}

/**
 * A grid PNG read back. It is a comparison sheet, not one picture's settings: the panel restores the
 * experiment or lets the user choose one cell, and applies nothing until that cell's dialog is used.
 */
export function XYZGridRecordPanel({ record, onOpenExperiment, promptContext, onApply, onGoBasic }) {
  const [detail, setDetail] = useState(null), [missing, setMissing] = useState(""), [cellId, setCellId] = useState(""), [applying, setApplying] = useState(false), [openError, setOpenError] = useState("");
  useEffect(() => {
    let stopped = false;
    setDetail(null); setMissing(""); setCellId(""); setOpenError("");
    if (!record?.experimentId) return undefined;
    api.detail(record.experimentId).then((value) => { if (!stopped) setDetail(value); })
      .catch((failure) => { if (!stopped) setMissing(failure.status === 404 ? "本服务没有这个实验（可能来自其他电脑或已被移走）。网格 PNG 不含单格参数；完整参数在同名 .json 清单中。" : failure.message); });
    return () => { stopped = true; };
  }, [record?.experimentId]);
  const digestMatches = detail && record?.definitionDigest ? detail.digest === record.definitionDigest : null;
  const cells = detail ? detail.cells.filter((cell) => !record.cellIds?.length || record.cellIds.includes(cell.id)) : [];
  const open = () => { const problem = onOpenExperiment?.({ id: record.experimentId, engine: detail?.engine, origin: record.search ? "lora_mix_search" : undefined }); if (problem) setOpenError(problem); };
  return <section className="xyz-grid-record" aria-label="XYZ 对比网格">
    <header><strong>{record.search ? `LoRA 混合搜索候选网格 · 第 ${record.search.round ?? "?"} 轮` : "XYZ 对比网格"}</strong><small>{record.search ? `${record.search.candidates ?? "?"} 个候选 · ${record.search.columns ?? "?"} 列 × ${record.search.rows ?? "?"} 行（显示布局，不是轴）；` : ""}不是单张图片的参数记录；读取网格不会自动应用任何参数</small></header>
    <dl>
      <div><dt>实验</dt><dd title={record.experimentId}>{record.experimentId}</dd></div>
      <div><dt>导出</dt><dd>{record.mode === "original" ? "原始分辨率" : `${record.percent}% 缩小`} · {record.width} × {record.height}{record.cellIds?.length ? ` · ${record.cellIds.length} 格` : ""}{record.overflow ? " · 格位置见清单" : ""}</dd></div>
      {detail && <div><dt>状态</dt><dd>{detail.engine} · {XYZ_EXPERIMENT_STATUS_TEXT[detail.status] || detail.status} · {detail.counts.complete}/{detail.total_cells} 格完成{digestMatches === false ? " · 定义摘要与网格不一致" : ""}</dd></div>}
    </dl>
    {missing && <p className="xyz-note" role="status">{missing}</p>}
    {detail && <>
      <div className="xyz-actions"><button type="button" onClick={open}>在 XYZ 工作区打开实验</button></div>
      {openError && <p role="alert">{openError}</p>}
      <label className="xyz-grid-record-cell">选择一格查看并应用<WorkspaceSelect ariaLabel="选择网格中的一格" value={cellId} onChange={setCellId}
        options={[{ value: "", label: "未选择" }, ...cells.map((cell) => ({ value: cell.id, label: `${cellTitle(cell)} · ${XYZ_CELL_STATUS_TEXT[cell.status] || cell.status}` }))]} /></label>
      <div className="xyz-actions"><button type="button" disabled={!cellId || !onApply} onClick={() => setApplying(true)}>查看此格参数并选择应用…</button></div>
    </>}
    {applying && cellId && <XYZCellApplyDialog experimentId={record.experimentId} cellId={cellId} promptContext={promptContext} onApply={onApply} onGoBasic={onGoBasic} onClose={() => setApplying(false)} />}
  </section>;
}
