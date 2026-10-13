import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, ImagePlus, ListOrdered, Maximize2, RefreshCw, WandSparkles } from "lucide-react";
import XYZCanvasPanel from "./XYZCanvasPanel.jsx";
import TaskDeletionManager from "./TaskDeletionManager.jsx";
import { removedExperiments } from "./task-deletion.js";
import { XYZCountField } from "./SamplingControls.jsx";
import { xyzWireDraft } from "./xyz-prompt-source.js";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import XYZAxisEditor from "./XYZAxisEditor.jsx";
import { useDialogLifecycle } from "./gallery-core.js";
import { VIEWER_MAX_CANVAS_EDGE } from "./viewer-editor.js";
import { XYZCellApplyDialog } from "./XYZCellActions.jsx";
import { xyzBasicPrompt, xyzCellGalleryOutput, xyzCellSettings, xyzViewerSource } from "./xyz-apply.js";
import { syncXYZBaseline, xyzLocalSummary } from "./xyz-draft.js";
import { xyzSidebarIssues } from "./xyz-sidebar-context.js";
import { xyzAxisRuntimeIssues } from "./xyz-axis-editor.js";
import {
  XYZ_CELL_STATUS_TEXT, XYZ_EXPERIMENT_STATUS_TEXT, XYZ_EXPORT_STATUS_TEXT, formatBytes, xyzApi, xyzCoordinateText, xyzErrorText,
  xyzExperimentActive, xyzExportActive, xyzExportBody, xyzGridPage, xyzIssues, xyzSubmissionId, xyzSubmissionKey,
} from "./xyz-client.js";

const api = xyzApi();
const syncLabels = { models: "模型 / 组件", prompts: "完整组合词 / 组快照", sampling: "采样 / 种子", canvas: "画布", loras: "LoRA 完整有序列表 / 身份", postprocess: "固定后处理链", output: "输出命名" };
const errorPolicies = [{ value: "pause", label: "暂停并等待选择（默认）" }, { value: "continue", label: "记录失败并继续其余格" }];
function issueList(issues, limit = 20) {
  return <ul className="xyz-issues">{issues.slice(0, limit).map((issue, index) => <li key={index}><b>{xyzCoordinateText(issue)}</b>{issue.labels ? ` · ${Object.values(issue.labels).join(" / ")}` : ""} · {issue.message}</li>)}
    {issues.length > limit && <li>…另有 {issues.length - limit} 项</li>}</ul>;
}

function SyncDialog({ baseline, draft, onSync, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const [fields, setFields] = useState(Object.keys(syncLabels));
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog panel" role="dialog" aria-modal="true" aria-labelledby="xyz-sync-title" ref={ref}>
    <h2 id="xyz-sync-title">从基础生图同步基线</h2>
    <p>仅覆盖勾选字段，X / Y / Z 与执行设置不变。不会应用参数、卸载模型或修改真实 LoRA 组。</p>
    {Object.entries(syncLabels).map(([key, label]) => <label key={key}><input type="checkbox" checked={fields.includes(key)} onChange={(event) => setFields((old) => event.target.checked ? [...old, key] : old.filter((item) => item !== key))} />{label}</label>)}
    <p>采样：{draft.baseline.parameters.steps} → {baseline.parameters.steps} 步 · 画布：{baseline.parameters.width} × {baseline.parameters.height} · 挂载：{draft.baseline.loras.length} → {baseline.loras.length}</p>
    <div className="xyz-actions"><button type="button" onClick={onClose}>取消</button><button type="button" disabled={!fields.length} onClick={() => { onSync(syncXYZBaseline(draft, baseline, fields)); onClose(); }}>同步所选字段</button></div>
  </section></div>, document.body);
}

function PlanSummary({ preview, stale }) {
  if (!preview) return null;
  return <section className={`xyz-plan-summary${preview.ok ? "" : " has-issues"}`} aria-live="polite">
    <h3>服务端预检{stale ? " · 草稿已修改，结果已过期" : ""}</h3>
    <p>{preview.dimensions.x} 列 × {preview.dimensions.y} 行 × {preview.dimensions.z} 网格 · 共 {preview.total_cells} 格 · 模型切换 {preview.switches.model} 次 · LoRA 组合切换 {preview.switches.loras} 次 · 种子{preview.seed_status === "resolved" ? "已固定" : "将在提交时抽取一次"}</p>
    {preview.batches && <p>预计 {preview.batches.count} 个执行批次（{Object.entries(preview.batches.sizes).map(([size, count]) => `${size} 张 × ${count}`).join("，")}）· 并发上限 {preview.requested_concurrency}。{preview.batches.reasons.slice(0, 2).join("；")}。显存策略或显存不足拆分可能让实际并发更低，届时如实显示。</p>}
    {preview.ok ? <p>每格都已通过与普通生成相同的提交前检查（模型 / LoRA 身份与字节 / 后处理资源）。真实权重与显存仍在执行时复核。</p>
      : <><p role="alert">{preview.error_count} 个问题需要先修正；提交会被拒绝且不抽取种子：</p>{issueList(preview.errors)}</>}
  </section>;
}

const CELL_SIZES = [{ value: "132", label: "小" }, { value: "180", label: "中" }, { value: "260", label: "大" }, { value: "380", label: "特大" }];

function ExperimentGrid({ detail, page, setPage, actions = {}, onExport }) {
  const zCount = detail.axes.z.labels.length || 1, samples = detail.repeat_count || 1;
  const [cellSize, setCellSize] = useState("180");
  const grid = useMemo(() => xyzGridPage(detail, page.z, page.sample), [detail, page]);
  const pages = [];
  for (let z = 0; z < zCount; z += 1) for (let sample = 0; sample < samples; sample += 1) pages.push({ z, sample });
  return <section className="xyz-grid-wrap" aria-label="实验网格" style={{ "--xyz-cell": `${cellSize}px` }}>
    <div className="xyz-grid-tools">
      {pages.length > 1 && <label className="xyz-grid-page"><span>子网格 / 样本</span><WorkspaceSelect ariaLabel="选择 Z 子网格与重复样本" value={`${page.z}:${page.sample}`}
        options={pages.map((item) => ({ value: `${item.z}:${item.sample}`, label: `Z${item.z + 1} ${detail.axes.z.labels[item.z] || ""} · 样本 ${item.sample + 1}` }))}
        onChange={(value) => { const [z, sample] = value.split(":").map(Number); setPage({ z, sample }); }} /></label>}
      <label className="xyz-grid-size"><span>屏幕格大小（不影响导出）</span><WorkspaceSelect ariaLabel="网格缩略图大小" value={cellSize} options={CELL_SIZES} onChange={setCellSize} /></label>
      <button type="button" className="xyz-export-entry" disabled={!detail.counts.complete} title={detail.counts.complete ? "导出整网格单张 PNG" : "至少需要 1 张已完成原图"} onClick={onExport}>导出整网格 PNG…</button>
    </div>
    <div className="xyz-grid-scroll"><table className="xyz-grid">
      <thead><tr><th scope="col"><span className="sr-only">Y \ X</span></th>{(detail.axes.x.labels.length ? detail.axes.x.labels : ["—"]).map((label, index) => <th scope="col" key={index} title={label}>{label || `X${index + 1}`}</th>)}</tr></thead>
      <tbody>{grid.map((row, y) => <tr key={y}><th scope="row" title={detail.axes.y.labels[y] || ""}>{detail.axes.y.labels[y] || `Y${y + 1}`}</th>
        {row.map((cell, x) => <td key={x}>{!cell ? null : <figure className={`xyz-cell ${cell.status}`} title={`${xyzCoordinateText({ coordinate: { x_index: cell.coordinate[0], y_index: cell.coordinate[1], z_index: cell.coordinate[2], sample_index: cell.coordinate[3] } })} · 种子 ${cell.seed}${cell.batch ? ` · 批次 ${cell.batch.size} 张（采样 ${cell.batch.effective} 张/次）` : ""}${cell.error ? ` · ${cell.error}` : ""}`}>
          {cell.status === "complete" && cell.asset_url
            ? actions.openViewer && cell.asset_id
              ? <button type="button" className="xyz-cell-open" title="在预览工作区打开原图" onClick={() => actions.openViewer(cell)}><img src={cell.thumbnail_url || cell.asset_url} alt={`X${x + 1} Y${y + 1} 原图`} loading="lazy" decoding="async" /></button>
              : <a href={cell.asset_url} target="_blank" rel="noreferrer" title="打开原图（新标签页）"><img src={cell.thumbnail_url || cell.asset_url} alt={`X${x + 1} Y${y + 1} 原图`} loading="lazy" decoding="async" /></a>
            : <span className="xyz-cell-placeholder">{XYZ_CELL_STATUS_TEXT[cell.status] || cell.status}{cell.attempts > 1 ? ` · 第 ${cell.attempts} 次` : ""}</span>}
          <figcaption>{XYZ_CELL_STATUS_TEXT[cell.status] || cell.status}{cell.width ? ` · 最终 ${cell.width}×${cell.height}` : ""}{cell.batch?.size > 1 ? ` · 批 ${cell.batch.index + 1}/${cell.batch.size}` : ""}</figcaption>
          <div className="xyz-cell-actions" role="group" aria-label={`${cell.id} 操作`}>
            {cell.asset_url && actions.openViewer && cell.asset_id && <button type="button" title="在预览工作区打开原图" aria-label={`预览 ${cell.id} 原图`} onClick={() => actions.openViewer(cell)}><Maximize2 size={12} /></button>}
            {cell.asset_url && actions.addToGallery && <button type="button" title="把这一格加入画廊（使用本格实际参数）" aria-label={`把 ${cell.id} 加入画廊`} disabled={actions.galleryBusy === cell.id} onClick={() => actions.addToGallery(cell)}><ImagePlus size={12} /></button>}
            {actions.apply && <button type="button" title="把这一格的参数应用到基础生图（逐组选择）" aria-label={`把 ${cell.id} 的参数应用到基础生图`} onClick={() => actions.apply(cell)}><WandSparkles size={12} /></button>}
            {cell.asset_url && <a className="xyz-cell-download" href={cell.asset_url} download={cell.output_name || true} title="下载原图" aria-label={`下载 ${cell.id} 原图`}><Download size={12} /></a>}
          </div>
        </figure>}</td>)}</tr>)}</tbody>
    </table></div>
  </section>;
}

const EXPORT_ARRANGEMENTS = [{ value: "vertical", label: "Z / 样本纵向排列" }, { value: "horizontal", label: "横向排列" }, { value: "grid", label: "自动网格" }];

/** A grid export opens in the Viewer only within the Viewer's own canvas edge; beyond it, download. */
function exportViewable(output) {
  return Boolean(output?.asset_id) && Math.max(output.width || 0, output.height || 0) <= VIEWER_MAX_CANVAS_EDGE;
}

/** `exportApi` / `extraBody` let the search workspace reuse this dialog for its candidate grids (same quality contract). */
export function ExportDialog({ detail, page, onClose, onFinished, onOpenViewer, exportApi = api, extraBody = null, extraControls = null, blockLabel = "Z / 样本块排列", captionLabel = "每格下方显示种子 / 状态" }) {
  const ref = useDialogLifecycle(true, onClose);
  const [options, setOptions] = useState({ scope: "all", mode: "original", percent: 100, theme: "light", arrangement: "vertical", captions: true });
  const [estimate, setEstimate] = useState(null), [problem, setProblem] = useState(null), [record, setRecord] = useState(null), [busy, setBusy] = useState(false);
  const body = { ...xyzExportBody(options, page), ...(extraBody || {}) }, bodyKey = JSON.stringify(body);
  const percentValid = options.mode === "original" || (Number.isInteger(options.percent) && options.percent >= 1 && options.percent <= 100);
  useEffect(() => {
    if (!percentValid) return undefined;
    let stopped = false;
    const timer = window.setTimeout(() => {
      exportApi.estimateExport(detail.id, JSON.parse(bodyKey))
        .then((value) => { if (!stopped) { setEstimate(value); setProblem(null); } })
        .catch((failure) => { if (!stopped) { setEstimate(null); setProblem({ message: failure.message, suggestions: failure.payload?.detail?.suggestions || [] }); } });
    }, 250);
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [detail.id, bodyKey, percentValid]);
  useEffect(() => {
    if (!xyzExportActive(record)) return undefined;
    let stopped = false;
    const timer = window.setTimeout(async () => {
      try { const value = await exportApi.exportStatus(detail.id, record.id); if (!stopped) { setRecord(value); if (!xyzExportActive(value)) onFinished?.(); } }
      catch (failure) { if (!stopped) setProblem({ message: failure.message, suggestions: [] }); }
    }, 800);
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [record, detail.id, onFinished]);
  const set = (patch) => setOptions((old) => ({ ...old, ...patch }));
  const sliderPercent = Number.isInteger(options.percent) ? Math.max(1, Math.min(100, options.percent)) : 50;
  const chooseMode = (mode) => set({ mode, percent: mode === "scaled" ? (options.mode === "scaled" ? options.percent : 50) : 100 });
  const start = async () => {
    setBusy(true); setProblem(null);
    try { setRecord((await exportApi.createExport(detail.id, body)).export); }
    catch (failure) { setProblem({ message: failure.message, suggestions: failure.payload?.detail?.suggestions || [] }); }
    finally { setBusy(false); }
  };
  const running = xyzExportActive(record), done = record && !running;
  const budget = estimate?.budget;
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog xyz-export-dialog panel" role="dialog" aria-modal="true" aria-labelledby="xyz-export-title" ref={ref}>
    <h2 id="xyz-export-title">导出整网格 PNG</h2>
    <p>从服务端原图生成一张完整网格 PNG，并附带完整 JSON 清单。导出不使用 GPU，不改动任何原图。</p>
    <fieldset disabled={running}><legend>范围</legend>
      <label className="xyz-check"><input type="radio" name="xyz-export-scope" checked={options.scope === "all"} onChange={() => set({ scope: "all" })} />整个实验（所有 Z 与样本，合成一张）</label>
      <label className="xyz-check"><input type="radio" name="xyz-export-scope" checked={options.scope === "page"} onChange={() => set({ scope: "page" })} />仅当前页（Z{(page?.z ?? 0) + 1} · 样本 {(page?.sample ?? 0) + 1}）</label>
    </fieldset>
    <fieldset disabled={running}><legend>分辨率</legend>
      <label className="xyz-check"><input type="radio" name="xyz-export-mode" checked={options.mode === "original"} onChange={() => chooseMode("original")} />原始分辨率（每张原图 1:1，不重采样、不放大）</label>
      <label className="xyz-check"><input type="radio" name="xyz-export-mode" checked={options.mode === "scaled"} onChange={() => chooseMode("scaled")} />等比例缩小（单次 Lanczos3 · 线性光 · 预乘 alpha）</label>
      {options.mode === "scaled" && <div className="xyz-export-percent">
        <input type="range" min="1" max="100" value={sliderPercent} aria-label="缩小百分比滑块" style={{ "--progress": `${sliderPercent}%` }} onChange={(event) => set({ percent: Number(event.target.value) })} />
        <label>百分比<input type="number" min="1" max="100" step="1" value={options.percent} aria-invalid={!percentValid} onChange={(event) => set({ percent: event.target.value === "" ? "" : Number(event.target.value) })} /></label>
        <div className="xyz-actions">{[25, 50, 75, 100].map((value) => <button type="button" key={value} aria-pressed={options.percent === value} onClick={() => set({ percent: value })}>{value}%</button>)}</div>
        {!percentValid && <p role="alert">请输入 1–100 的整数百分比</p>}
      </div>}
    </fieldset>
    <fieldset disabled={running} className="xyz-export-look"><legend>版式</legend>
      <label>报告主题<WorkspaceSelect ariaLabel="导出主题" value={options.theme} options={[{ value: "light", label: "浅色（默认）" }, { value: "dark", label: "深色" }]} onChange={(theme) => set({ theme })} /></label>
      {extraControls}
      <label>{blockLabel}<WorkspaceSelect ariaLabel="导出 Z 块排列" value={options.arrangement} options={EXPORT_ARRANGEMENTS} onChange={(arrangement) => set({ arrangement })} /></label>
      <label className="xyz-check"><input type="checkbox" checked={options.captions} onChange={(event) => set({ captions: event.target.checked })} />{captionLabel}</label>
    </fieldset>
    <section className={`xyz-export-estimate${budget && !budget.ok ? " over" : ""}`} aria-live="polite">
      {!percentValid ? <p role="alert">请输入 1–100 的整数百分比；导出已暂停。</p> : estimate ? <>
        <p><strong>{estimate.width} × {estimate.height} px</strong> · {(estimate.width * estimate.height / 1e6).toFixed(1)} MP · {estimate.percent}% · 预计内存 {formatBytes(budget.memory_bytes)} · {budget.ok ? "预算内" : "超出预算"}</p>
        <p>{estimate.blocks} 个子网格 · {Object.entries(estimate.cells).map(([status, count]) => `${XYZ_CELL_STATUS_TEXT[status] || status} ${count}`).join(" · ")}{estimate.cells.complete !== Object.values(estimate.cells).reduce((a, b) => a + b, 0) ? " · 未完成格以占位显示" : ""}</p>
        {!budget.ok && <p role="alert">{budget.problems.join("；")}。不会自动缩小、分页或省略。</p>}
        {estimate.warnings.map((warning) => <p key={warning.code} className="xyz-warning">{warning.message}</p>)}
        {!budget.ok && estimate.suggestions.length > 0 && <div className="xyz-actions"><span>可选比例：</span>{estimate.suggestions.map((value) => <button type="button" key={value} onClick={() => set({ mode: "scaled", percent: value })}>{value}%</button>)}</div>}
      </> : problem ? <p role="alert">{problem.message}</p> : <p>正在计算最终尺寸与预算…</p>}
      {problem && estimate && <p role="alert">{problem.message}</p>}
      {problem?.suggestions?.length > 0 && <div className="xyz-actions"><span>可选比例：</span>{problem.suggestions.map((value) => <button type="button" key={value} onClick={() => set({ mode: "scaled", percent: value })}>{value}%</button>)}</div>}
    </section>
    {record && <section className="xyz-export-progress" aria-live="polite">
      <p>{XYZ_EXPORT_STATUS_TEXT[record.status] || record.status} · {record.progress?.done ?? 0}/{record.progress?.total ?? 0} 张原图{record.error ? ` · ${record.error}` : ""}</p>
      <div className="xyz-progress" role="progressbar" aria-valuemin={0} aria-valuemax={record.progress?.total || 1} aria-valuenow={record.progress?.done || 0}><i style={{ width: `${record.progress?.total ? Math.round(100 * record.progress.done / record.progress.total) : 0}%` }} /></div>
      {record.status === "complete" && <p>{onOpenViewer && exportViewable(record.output) && <><button type="button" className="xyz-link-button" onClick={() => { onOpenViewer(xyzViewerSource({ assetId: record.output.asset_id, name: record.output.name })); onClose(); }}>在预览工作区打开</button> · </>}<a href={record.output.asset_url} target="_blank" rel="noreferrer">打开 PNG</a> ·<a href={record.output.asset_url} download={record.output.name}>下载 PNG（{formatBytes(record.output.bytes)}）</a> · <a href={record.manifest_url} download>下载 JSON 清单</a>{record.metadata_overflow ? " · PNG 内元数据超出预算，完整信息见清单" : ""}</p>}
    </section>}
    <div className="xyz-actions">
      {running ? <button type="button" onClick={() => exportApi.cancelExport(detail.id, record.id).catch((failure) => setProblem({ message: failure.message, suggestions: [] }))}>取消导出</button>
        : <button type="button" onClick={onClose}>{done ? "关闭" : "取消"}</button>}
      {!done && <button type="button" className="primary" disabled={busy || running || !estimate || !budget?.ok || !percentValid} onClick={start}>{busy ? "正在提交…" : "导出 PNG"}</button>}
    </div>
  </section></div>, document.body);
}

function ExperimentPanel({ detail, onAction, busy, maxConcurrency, onRefresh, actions = {} }) {
  const [page, setPage] = useState({ z: 0, sample: 0 }), [exporting, setExporting] = useState(false);
  const [limit, setLimit] = useState(String(detail.requested_concurrency || 1));
  const limitValue = Number(limit), limitValid = Number.isInteger(limitValue) && limitValue >= 1 && limitValue <= maxConcurrency;
  const counts = detail.counts, total = detail.total_cells;
  const finished = counts.complete + counts.error + counts.skipped + counts.cancelled + counts.missing;
  const active = xyzExperimentActive(detail), decision = detail.decision;
  const failed = counts.error + counts.skipped + counts.missing;
  const unfinished = counts.pending + counts.cancelled;
  const live = detail.live;
  const when = detail.created_at ? new Date(detail.created_at * 1000).toLocaleString() : "所选实验";
  // The server's reason often repeats "请求 N / 实际 M：…"; the header already states both numbers.
  const reason = String(detail.concurrency_reason || "").replace(/^请求\s*\d+\s*\/\s*(实际|分组|本批)[^：:]*[：:]\s*/, "").trim();
  const liveText = live?.phase ? `${live.phase}${live.cell_progress != null ? ` · 当前格 ${live.cell_progress}%` : ""}` : detail.status === "paused" ? "已暂停 · 可继续生成或停止" : active ? "等待服务端状态…" : "";
  const controls = <footer className={`xyz-run-bar xyz-result-run-bar${!active && unfinished === 0 && failed === 0 ? " xyz-result-complete" : ""}`} aria-label="选中实验运行操作">
    <div><strong title={`实验 ${detail.id}`}>{when} · {XYZ_EXPERIMENT_STATUS_TEXT[detail.status] || detail.status}</strong><p>{counts.complete}/{total} 格完成 · 请求 {detail.requested_concurrency} / 实际 {detail.effective_concurrency ?? "—"}{detail.foreign ? " · 其他进程拥有，只读" : " · 控制仅作用于此实验"}</p></div>
    <div className="xyz-run-actions">
      {active && !decision && ["queued", "running"].includes(detail.status) && <button type="button" disabled={busy || detail.foreign} onClick={() => onAction("control", "pause")}>暂停</button>}
      {active && !decision && ["pausing", "paused"].includes(detail.status) && <button type="button" disabled={busy || detail.foreign} onClick={() => onAction("control", "resume")}>{detail.status === "paused" ? "继续生成" : "取消暂停并继续"}</button>}
      {active && !decision && detail.status !== "cancelling" && <button type="button" disabled={busy || detail.foreign} onClick={() => onAction("control", "stop")}>停止（保留已完成原图）</button>}
      {!active && !detail.foreign && (unfinished > 0 || failed > 0) && <label className="xyz-continue-limit">继续时的并发上限<input type="number" min="1" max={maxConcurrency} step="1" value={limit} aria-invalid={!limitValid} onChange={(event) => setLimit(event.target.value)} /></label>}
      {!active && !detail.foreign && unfinished > 0 && <button type="button" disabled={busy || !limitValid} onClick={() => onAction("continue", "none", limitValue)}>继续未完成的 {unfinished} 格</button>}
      {!active && !detail.foreign && failed > 0 && <button type="button" disabled={busy || !limitValid} onClick={() => onAction("continue", "failed", limitValue)}>重试失败 / 缺失的 {failed} 格（生成新文件）</button>}
      {!active && unfinished === 0 && failed === 0 && <span className="xyz-note">已完成 · 可查看 / 导出</span>}
    </div>
  </footer>;
  return <section className="xyz-experiment" aria-label="实验状态">
    <div className="xyz-experiment-main">
    <details className="xyz-frozen-summary"><summary><strong>冻结基线 · {detail.engine}</strong><span>原始 {detail.baseline?.width ?? "—"} × {detail.baseline?.height ?? "—"} · {detail.baseline?.steps ?? "—"} 步</span></summary><p>{detail.baseline?.checkpoint || detail.baseline?.diffusion_model || "服务端实验记录"} · {detail.baseline?.loras ?? "—"} LoRA</p><small title={detail.definition_digest || detail.digest}>实验 {detail.id} · 不读取当前草稿</small></details>
    <header className="xyz-experiment-head">
      <div><strong>{XYZ_EXPERIMENT_STATUS_TEXT[detail.status] || detail.status}</strong><span> · {counts.complete}/{total} 格完成{counts.error ? ` · ${counts.error} 失败` : ""}{counts.skipped ? ` · ${counts.skipped} 跳过` : ""}{counts.missing ? ` · ${counts.missing} 原图缺失` : ""}</span></div>
      <small title={detail.concurrency_reason || undefined}>请求并发 {detail.requested_concurrency} / 实际 {detail.effective_concurrency ?? "—"}{reason ? ` · ${reason}` : ""}</small>
    </header>
    <div className="xyz-progress" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={finished}><i style={{ width: `${total ? Math.round(100 * finished / total) : 0}%` }} /></div>
    {liveText && <p className="xyz-live" aria-live="polite">{liveText}</p>}
    {detail.foreign && <p role="alert">另一个仍在运行的服务进程拥有此实验；此页面只读。</p>}
    {decision && <div className="xyz-decision" role="alert"><p>格 {decision.cell_id} 失败：{decision.error}</p><div className="xyz-actions">
      <button type="button" disabled={busy} onClick={() => onAction("decide", "retry")}>重试此格（同一冻结请求与种子）</button>
      <button type="button" disabled={busy} onClick={() => onAction("decide", "skip")}>跳过此格并继续</button>
      <button type="button" disabled={busy} onClick={() => onAction("decide", "stop")}>停止实验</button></div></div>}
    {detail.error && <p role="alert">{detail.error}</p>}
    <ExperimentGrid key={detail.id} detail={detail} page={page} setPage={setPage} actions={actions} onExport={() => setExporting(true)} />
    {detail.exports?.length > 0 && <details className="xyz-export-list"><summary>导出记录 · {detail.exports.length} 条</summary>
      <ol>{detail.exports.slice().reverse().map((record) => <li key={record.id}><b>{XYZ_EXPORT_STATUS_TEXT[record.status] || record.status}</b> · {record.options?.scope === "page" ? "当前页" : "整个实验"} · {record.options?.mode === "original" ? "原始分辨率" : `${record.options?.percent ?? "—"}%`}
        {record.output ? <> · {record.output.width}×{record.output.height} · {actions.openViewer && exportViewable(record.output) && <><button type="button" className="xyz-link-button" onClick={() => actions.openViewer({ asset_id: record.output.asset_id, output_name: record.output.name })}>预览</button> · </>}<a href={record.output.asset_url} target="_blank" rel="noreferrer">打开</a> ·<a href={record.output.asset_url} download={record.output.name}>下载</a> · <a href={record.manifest_url} download>清单</a></> : record.error ? ` · ${record.error}` : ""}</li>)}</ol></details>}
    {exporting && <ExportDialog detail={detail} page={page} onClose={() => setExporting(false)} onFinished={onRefresh} onOpenViewer={actions.openViewerSource} />}
    {detail.batch_log?.length > 0 && <details className="xyz-batch-log"><summary>执行批次 · {detail.batch_log.length} 条（只合并仅种子不同的格）</summary>
      <ol>{detail.batch_log.slice().reverse().map((batch, index) => <li key={`${batch.at}-${index}`}><b>{batch.size} 张{batch.sampling_batch_size && batch.sampling_batch_size < batch.size ? ` · 采样每批 ${batch.sampling_batch_size} 张` : ""}</b> · {XYZ_CELL_STATUS_TEXT[batch.status] || batch.status}{batch.oom ? " · 显存不足" : ""} · {batch.cells.join("、")} · {batch.reason}</li>)}</ol></details>}
    {detail.cells.some((cell) => cell.error) && <details className="xyz-cell-errors"><summary>失败与说明</summary>
      <ul>{detail.cells.filter((cell) => cell.error).map((cell) => <li key={cell.id}><b>{cell.id}</b> · {Object.values(cell.labels).join(" / ")} · {cell.error}</li>)}</ul></details>}
    <p className="xyz-note">原图逐格单独保存（批次中每张图也各自对应自己的格与种子），并保留普通单图参数、画廊与工作流。屏幕网格使用缩略图；导出始终读取服务端原图。</p>
    </div>
    {controls}
  </section>;
}

function ExperimentResults({ engine, selectedId, onSelect, refreshToken, maxConcurrency, integration = {} }) {
  const [listing, setListing] = useState(null), [detail, setDetail] = useState(null);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [applyCell, setApplyCell] = useState(""), [galleryBusy, setGalleryBusy] = useState("");
  const listRequest = useRef(0), detailRequest = useRef(0), galleryRequest = useRef(0);
  const selectedRef = useRef(selectedId), mountedRef = useRef(true);
  selectedRef.current = selectedId;
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; listRequest.current += 1; detailRequest.current += 1; galleryRequest.current += 1; }; }, []);
  const detailId = detail?.id;
  // Viewer, Gallery and Apply reach the owners App already has. A cell's Gallery card and Apply use
  // that cell's frozen request, never the draft on screen.
  const actions = useMemo(() => ({
    openViewerSource: integration.onOpenViewer || null,
    openViewer: integration.onOpenViewer ? (cell) => integration.onOpenViewer(xyzViewerSource({ assetId: cell.asset_id, name: cell.output_name })) : null,
    apply: integration.onApplyToBasic ? (cell) => setApplyCell(cell.id) : null,
    addToGallery: integration.onAddToGallery && detailId ? async (cell) => {
      const requestId = ++galleryRequest.current, experimentId = detailId;
      setGalleryBusy(cell.id);
      try {
        const full = await api.cell(experimentId, cell.id);
        if (!mountedRef.current || requestId !== galleryRequest.current || selectedRef.current !== experimentId) return;
        const context = integration.promptContext || {};
        const prompt = xyzBasicPrompt(full.prompt_parts, { ...context, groupPrompt: context.groupPromptFor ? context.groupPromptFor(null) : "" });
        integration.onAddToGallery({ outputs: [xyzCellGalleryOutput({ ...full, thumbnail_url: cell.thumbnail_url })], settings: xyzCellSettings(full, prompt) });
      } catch (failure) { if (mountedRef.current && requestId === galleryRequest.current && selectedRef.current === experimentId) setError(failure.message); }
      finally { if (mountedRef.current && requestId === galleryRequest.current) setGalleryBusy(""); }
    } : null,
    galleryBusy,
  }), [integration, detailId, galleryBusy]);
  const loadList = useCallback(async () => {
    const requestId = ++listRequest.current;
    try {
      const value = await api.list(engine);
      if (mountedRef.current && requestId === listRequest.current) { setListing(value); setError(""); return value; }
    } catch (failure) { if (mountedRef.current && requestId === listRequest.current) setError(failure.message); }
  }, [engine]);
  const loadDetail = useCallback(async (id) => {
    const requestId = ++detailRequest.current;
    if (!id) { if (mountedRef.current && selectedRef.current === id) setDetail(null); return null; }
    try {
      const value = await api.detail(id);
      if (mountedRef.current && requestId === detailRequest.current && selectedRef.current === id) { setDetail(value); setError(""); return value; }
      return null;
    } catch (failure) { if (mountedRef.current && requestId === detailRequest.current && selectedRef.current === id) setError(failure.message); return null; }
  }, []);
  useEffect(() => { void loadList(); return () => { listRequest.current += 1; }; }, [loadList, refreshToken]);
  useEffect(() => { detailRequest.current += 1; galleryRequest.current += 1; setDetail(null); setApplyCell(""); setGalleryBusy(""); setBusy(false); setError(""); }, [selectedId, engine]);
  // The server filters by origin; this also keeps a not-yet-restarted service from mixing search rounds into XYZ.
  const experiments = useMemo(() => (listing?.experiments || []).filter((item) => !item.origin), [listing]);
  useEffect(() => {
    if (!selectedId && experiments.length) onSelect(experiments[0].id);
  }, [experiments, selectedId, onSelect]);
  const detailActive = xyzExperimentActive(detail), detailStatus = detail?.status;
  // The picker's summaries follow every status change of the selected experiment.
  useEffect(() => { if (detailStatus) void loadList(); }, [detailStatus, loadList]);
  // Server snapshots only. Polling stops once the selected experiment is no longer active.
  useEffect(() => {
    let stopped = false, timer = 0;
    const tick = async () => {
      const value = await loadDetail(selectedId);
      if (!stopped && xyzExperimentActive(value)) timer = window.setTimeout(tick, 1500);
      else if (!stopped) void loadList();
    };
    void tick();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [selectedId, loadDetail, loadList, refreshToken, detailActive]);
  const act = async (kind, value, limit) => {
    if (!detail?.id || busy) return;
    const experimentId = detail.id;
    setBusy(true);
    try {
      if (kind === "control") await api.control(experimentId, value);
      else if (kind === "decide") await api.decide(experimentId, value);
      else await api.continue(experimentId, value, { requestedConcurrency: limit });
      if (mountedRef.current && selectedRef.current === experimentId) { await loadDetail(experimentId); await loadList(); }
    } catch (failure) { if (mountedRef.current && selectedRef.current === experimentId) setError(failure.message); }
    finally { if (mountedRef.current && selectedRef.current === experimentId) setBusy(false); }
  };
  const refreshDetail = useCallback(() => { if (detail?.id) void loadDetail(detail.id); }, [detail?.id, loadDetail]);
  const otherActive = experiments.filter((item) => item.id !== selectedId && xyzExperimentActive(item));
  const deleted = async (result) => {
    const ids = removedExperiments(result);
    setListing((old) => old ? { ...old, experiments: old.experiments.filter((item) => !ids.includes(item.id)) } : old);
    if (ids.includes(selectedRef.current)) {
      detailRequest.current++; galleryRequest.current++; setDetail(null); setApplyCell(""); setGalleryBusy("");
      onSelect(experiments.find((item) => !ids.includes(item.id))?.id || "");
    }
    await loadList();
  };
  // The task manager keeps one position in the header bar (with or without experiments) so it never remounts mid-deletion.
  return <section className="xyz-results" aria-label="XYZ 实验结果">
    <div className="xyz-results-bar">
      {experiments.length ? <label className="xyz-experiment-pick">实验<WorkspaceSelect ariaLabel="选择 XYZ 实验" value={selectedId || ""} onChange={onSelect}
        options={experiments.map((item) => ({ value: item.id, label: `${new Date(item.created_at * 1000).toLocaleString()} · ${XYZ_EXPERIMENT_STATUS_TEXT[item.status] || item.status} · ${item.counts.complete}/${item.total_cells} 格` }))} /></label>
        : <p className="xyz-results-empty-text" role="status">{listing ? "此引擎还没有 XYZ 实验。提交后关闭浏览器也会继续执行，回来即可恢复查看。" : "正在读取服务端实验记录…"}</p>}
      <TaskDeletionManager api={api} module="xyz" engine={engine} refreshToken={refreshToken} onDeleted={deleted} currentId={selectedId} />
    </div>
    {error && <p role="alert" className="xyz-error">{error}</p>}
    {experiments.length > 0 && <>
      {otherActive.length > 0 && <div className="xyz-other-active" role="status"><span>另有活动实验：</span>{otherActive.map((item) => <button type="button" key={item.id} title={`实验 ${item.id}`} onClick={() => onSelect(item.id)}>查看 {new Date(item.created_at * 1000).toLocaleString()} · {XYZ_EXPERIMENT_STATUS_TEXT[item.status] || item.status} · {item.counts.complete}/{item.total_cells} 格</button>)}</div>}
      {detail && detail.id === selectedId && detail.origin ? <p className="xyz-note" role="status">此实验属于其他工作区（{detail.origin.kind === "lora_mix_search" ? "LoRA 混合搜索" : detail.origin.kind || "未知来源"}），请在其结果页查看；XYZ 结果只列出 XYZ 实验。</p>
        : detail && detail.id === selectedId ? <ExperimentPanel key={detail.id} detail={detail} busy={busy} onAction={act} maxConcurrency={maxConcurrency} onRefresh={refreshDetail} actions={actions} /> : <p role="status">正在读取实验…</p>}
    </>}
    {applyCell && detailId && <XYZCellApplyDialog experimentId={detailId} cellId={applyCell} promptContext={integration.promptContext} onApply={integration.onApplyToBasic} onGoBasic={integration.onGoBasic} onClose={() => setApplyCell("")} />}
    {listing?.unreadable?.length > 0 && <p className="xyz-note">{listing.unreadable.length} 条实验记录无法安全读取，原始文件已保留未修改。</p>}
  </section>;
}

export default function XYZPlotWorkspace({ draft, onChange, captureBaseline, error, engine, focusExperiment, onFocusHandled, queueOpen, queueOutstanding, onToggleQueue, integration, componentChoices = {}, runtimeGuidance, promptDeck, canvasMaxEdge = 2048, loraManager }) {
  const [sync, setSync] = useState(null), [tab, setTab] = useState("config");
  const [capabilities, setCapabilities] = useState(null), [onError, setOnError] = useState("pause");
  const [preview, setPreview] = useState(null), [previewKey, setPreviewKey] = useState(""), [previewRuntimeGuidanceKey, setPreviewRuntimeGuidanceKey] = useState(""), [planBusy, setPlanBusy] = useState(false), [planError, setPlanError] = useState(null);
  const [submitBusy, setSubmitBusy] = useState(false), [submitError, setSubmitError] = useState(null), [submittedNotice, setSubmittedNotice] = useState(null);
  const [selectedId, setSelectedId] = useState(""), [refreshToken, setRefreshToken] = useState(0);
  const pendingSubmission = useRef(null), submitFlight = useRef(false), mounted = useRef(true), planRequest = useRef(0);
  const axisPreviousStates = useRef({ x: new Map(), y: new Map(), z: new Map() });
  const draftKey = useMemo(() => {if(!draft)return '';try{return JSON.stringify(xyzWireDraft(draft));}catch{return `invalid:${JSON.stringify(draft)}`;}}, [draft]);
  const runtimeGuidanceKey = JSON.stringify(runtimeGuidance ?? null);
  const draftKeyRef = useRef(draftKey), engineRef = useRef(engine || draft?.engine), lastDraftKey = useRef(draftKey);
  const runtimeGuidanceKeyRef = useRef(runtimeGuidanceKey), lastRuntimeGuidanceKey = useRef(runtimeGuidanceKey);
  draftKeyRef.current = draftKey;
  engineRef.current = engine || draft?.engine;
  runtimeGuidanceKeyRef.current = runtimeGuidanceKey;
  useEffect(() => {
    let stopped = false;
    api.capabilities().then((value) => { if (!stopped) setCapabilities(value); })
      .catch(() => { if (!stopped) setCapabilities({ executor_ready: false, max_concurrency: 1 }); });
    return () => { stopped = true; };
  }, []);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; planRequest.current += 1; }; }, []);
  useEffect(() => {
    if (lastDraftKey.current === draftKey) return;
    lastDraftKey.current = draftKey;
    planRequest.current += 1;
    setPlanBusy(false);
    setPlanError(null);
    setSubmitError(null);
    setSubmittedNotice(null);
  }, [draftKey]);
  useEffect(() => {
    if (lastRuntimeGuidanceKey.current === runtimeGuidanceKey) return;
    lastRuntimeGuidanceKey.current = runtimeGuidanceKey;
    planRequest.current += 1;
    setPlanBusy(false);
    setPlanError(null);
  }, [runtimeGuidanceKey]);
  // The server's maximum is the only authority; the draft keeps whatever the user typed.
  useEffect(() => {
    if (!focusExperiment) return;
    setTab("results"); setSelectedId(focusExperiment); onFocusHandled?.();
  }, [focusExperiment, onFocusHandled]);
  if (error) return <section className="xyz-workspace panel"><h2>XYZ 草稿无法恢复</h2><p role="alert">{error}</p><p>原始草稿未删除；请先检查数据版本。不会自动从 basic 重建覆盖。</p></section>;
  if (!draft) return <section className="xyz-workspace panel" role="status">等待当前引擎基线就绪…</section>;
  const summary = xyzLocalSummary(draft), baseline = draft.baseline.parameters;
  const maxConcurrency = Math.max(1, Math.min(10, capabilities?.max_concurrency ?? 1));
  const concurrencyError = Number.isInteger(draft.requested_concurrency) && draft.requested_concurrency > maxConcurrency
    ? `服务端最大并发为 ${maxConcurrency}；保留当前输入，提交会被拒绝，不会自动降低` : "";
  const baselineIssues = xyzSidebarIssues(draft);
  const axisRuntimeIssues = xyzAxisRuntimeIssues(draft, runtimeGuidance);
  const runPlan = async () => {
    if (planBusy || !summary.valid || baselineIssues.length || concurrencyError || axisRuntimeIssues.length) return;
    const requestId = ++planRequest.current, submittedKey = draftKey, submittedEngine = engine || draft.engine, submittedRuntimeKey = runtimeGuidanceKey, snapshot = structuredClone(draft);
    setPlanBusy(true); setPlanError(null);
    try {
      const value = await api.plan(snapshot);
      if (mounted.current && requestId === planRequest.current && draftKeyRef.current === submittedKey && engineRef.current === submittedEngine && runtimeGuidanceKeyRef.current === submittedRuntimeKey) { setPreview(value); setPreviewKey(submittedKey); setPreviewRuntimeGuidanceKey(submittedRuntimeKey); }
    } catch (failure) {
      if (mounted.current && requestId === planRequest.current && draftKeyRef.current === submittedKey && engineRef.current === submittedEngine && runtimeGuidanceKeyRef.current === submittedRuntimeKey) setPlanError({ key: submittedKey, runtimeKey: submittedRuntimeKey, message: failure.message, issues: xyzIssues(failure.payload) });
    } finally { if (mounted.current && requestId === planRequest.current) setPlanBusy(false); }
  };
  const start = async () => {
    if (submitFlight.current || !capabilities?.executor_ready || !summary.valid || baselineIssues.length || concurrencyError || axisRuntimeIssues.length) return;
    const submittedKey = draftKey, submittedEngine = engine || draft.engine, policy = onError, snapshot = structuredClone(draft);
    const submissionKey = xyzSubmissionKey(submittedKey, policy);
    if (!pendingSubmission.current || pendingSubmission.current.key !== submissionKey) pendingSubmission.current = { id: xyzSubmissionId(), key: submissionKey };
    const submissionId = pendingSubmission.current.id;
    submitFlight.current = true;
    setSubmitBusy(true); setSubmitError(null); setSubmittedNotice(null);
    try {
      const result = await api.submit(snapshot, { submissionId, onError: policy });
      if (!mounted.current || engineRef.current !== submittedEngine) return;
      if (pendingSubmission.current?.id === submissionId) pendingSubmission.current = null;
      if (draftKeyRef.current === submittedKey) { setSelectedId(result.experiment.id); setRefreshToken((value) => value + 1); setTab("results"); }
      else setSubmittedNotice({ id: result.experiment.id, key: submittedKey });
    } catch (failure) {
      // A definitive server answer retires the id; only a transport failure may resend it.
      if (!mounted.current || engineRef.current !== submittedEngine) return;
      if (failure.status && pendingSubmission.current?.id === submissionId) pendingSubmission.current = null;
      if (draftKeyRef.current === submittedKey) setSubmitError({ key: submittedKey, message: failure.message, issues: xyzIssues(failure.payload) });
    } finally {
      if (mounted.current && engineRef.current === submittedEngine) setSubmitBusy(false);
      submitFlight.current = false;
    }
  };
  const canStart = capabilities?.executor_ready && summary.valid && !concurrencyError && !submitBusy && !baselineIssues.length && !axisRuntimeIssues.length;
  const stale=Boolean(preview)&&(previewKey!==draftKey||previewRuntimeGuidanceKey!==runtimeGuidanceKey);
  const runState=planBusy?'loading':submitBusy?'submitting':summary.error||baselineIssues.length||axisRuntimeIssues.length||concurrencyError||planError?.key===draftKey||submitError?.key===draftKey?'error':stale?'stale':preview?.ok?'ready':'unchecked';
  const workspaceToolbar = <div className="xyz-toolbar">
    <div className="xyz-tabs" role="group" aria-label="XYZ 工作区视图"><button type="button" aria-pressed={tab === "config"} onClick={() => setTab("config")}>配置</button><button type="button" aria-pressed={tab === "results"} onClick={() => setTab("results")}>结果</button></div>
    <div className="xyz-head-actions">{onToggleQueue && <button type="button" className={queueOutstanding ? "busy" : ""} aria-expanded={Boolean(queueOpen)} title="查看任务队列" onClick={onToggleQueue}><ListOrdered size={14} />队列</button>}<button type="button" disabled={tab !== "config" || !captureBaseline || submitBusy} title={tab === "results" ? "返回配置后同步；不会改写已提交实验" : "选择要同步的基础生图字段"} onClick={() => setSync(captureBaseline())}><RefreshCw size={14} />同步基础基线</button></div>
  </div>;
  return <section className="xyz-workspace panel" aria-label="XYZ 对比工作区">
    {workspaceToolbar}
    {tab === "config" ? <div className="xyz-workspace-scroll">
      {promptDeck}
      <div className="xyz-canvas-axes">
      <XYZCanvasPanel draft={draft} onChange={onChange} canvasMaxEdge={canvasMaxEdge} />
      <div className="xyz-axes" aria-label="X 列、Y 行与 Z 子网格参数">{["x", "y", "z"].map((key) => <XYZAxisEditor key={`${draft.engine}:${key}`} axisKey={key} draft={draft} previousStates={axisPreviousStates.current[key]} componentChoices={componentChoices} runtimeGuidance={runtimeGuidance} loraManager={loraManager} onChange={(axis) => onChange({ ...draft, axes: { ...draft.axes, [key]: axis } })} />)}</div>
      </div>
      {axisRuntimeIssues.length > 0 && <div className="xyz-submit-error" role="alert" aria-label="运行时引导能力校验">{axisRuntimeIssues.map((issue, index) => <p key={`${issue}:${index}`}>{issue}。草稿原值保留；读取到兼容能力或明确修改轴后才可预检。</p>)}</div>}
      <div className="xyz-config-options">
        <div className="xyz-counts xyz-mother-controls"><XYZCountField label="重复样本" detail="每个参数组合的重复次数" value={draft.repeat_count} max={32} invalid={!Number.isInteger(draft.repeat_count) || draft.repeat_count < 1 || draft.repeat_count > 32} onChange={(repeat_count) => onChange({ ...draft, repeat_count })} />
        <XYZCountField label="请求并发" detail={`服务端上限 ${capabilities?.max_concurrency ?? "读取中"}；不增加格数`} value={draft.requested_concurrency} max={maxConcurrency} invalid={Boolean(concurrencyError) || !Number.isInteger(draft.requested_concurrency) || draft.requested_concurrency < 1} onChange={(requested_concurrency) => onChange({ ...draft, requested_concurrency })} /></div>
        <details className="xyz-execution-settings"><summary>执行设置 · 请求并发 {draft.requested_concurrency || "?"}</summary>
          <small>实际批次与原因以服务端结果为准；重复样本与并发独立。</small>
          <label>单格失败策略<WorkspaceSelect ariaLabel="XYZ 单格失败策略" value={onError} options={errorPolicies} onChange={setOnError} /></label>
          <label className="xyz-check"><input type="checkbox" checked={draft.group_by_resources === true} onChange={(event) => onChange({ ...draft, group_by_resources: event.target.checked })} />按模型 / LoRA 分组执行（只改顺序，不改坐标）</label>
        </details>
      </div>
      <PlanSummary preview={preview} stale={Boolean(preview) && (previewKey !== draftKey || previewRuntimeGuidanceKey !== runtimeGuidanceKey)} />
      {planError?.key === draftKey && planError.runtimeKey === runtimeGuidanceKey && <div className="xyz-submit-error" role="alert"><p>预检失败 · {planError.message}</p>{planError.issues.length > 0 && issueList(planError.issues)}</div>}
      {submitError?.key === draftKey && <div className="xyz-submit-error" role="alert"><p>提交失败 · {submitError.message}</p>{submitError.issues.length > 0 && issueList(submitError.issues)}</div>}
    </div> : <>
      <div className="xyz-results-context"><strong>已提交实验是服务端冻结快照</strong><span>它与「{draft.engine} · 新草稿」相互独立；编辑不会改写旧格参数、状态或原图。</span></div>
      <ExperimentResults engine={engine || draft.engine} selectedId={selectedId} onSelect={setSelectedId} refreshToken={refreshToken} maxConcurrency={maxConcurrency} integration={integration} />
    </>}
    {submittedNotice && <div className="xyz-submit-notice" role="status">较早版本草稿已成功提交为独立实验。<button type="button" onClick={() => { setSelectedId(submittedNotice.id); setRefreshToken((value) => value + 1); setTab("results"); setSubmittedNotice(null); }}>查看该实验</button></div>}
     {tab === "config" && <footer className="xyz-run-bar xyz-draft-run-bar" data-state={runState} aria-label="当前新草稿运行操作"><div aria-live="polite"><strong className="xyz-run-total">{summary.total === null ? '待定义' : `${summary.total} 张`}</strong><span className="xyz-run-detail">{summary.total === null ? '新草稿 · 组合计数无法完成' : `${summary.dimensions.x} 列 × ${summary.dimensions.y} 行 × ${summary.dimensions.z} 子网格 · ${draft.repeat_count} 次重复`}</span><p className="xyz-run-status">{summary.error || baselineIssues[0] || axisRuntimeIssues[0] || concurrencyError || (planError?.key===draftKey ? `预检失败 · ${planError.message}` : submitError?.key===draftKey ? `提交失败 · ${submitError.message}` : planBusy?'正在服务端预检…':submitBusy?'正在提交冻结草稿…':stale ? "预检过期：草稿或运行时能力已变更，请重新预检。" : !capabilities ? "正在读取执行服务能力…" : !capabilities.executor_ready ? "执行服务当前不可用；草稿与旧结果仍可浏览。" : preview?.ok ? "预检通过 · 执行时仍校验真实权重与资源。" : preview && !preview.ok ? '预检未通过 · 请修复上方问题。' : "尚未预检 · 本地计数不是服务端准入。")}</p></div>
       <div className="xyz-run-actions"><button type="button" disabled={planBusy || !summary.valid || baselineIssues.length > 0 || Boolean(concurrencyError) || submitBusy || axisRuntimeIssues.length > 0} onClick={runPlan}>{planBusy ? "正在预检…" : "服务端预检"}</button>
         <button type="button" className="primary" disabled={!canStart} title={capabilities && !capabilities.executor_ready ? "推理服务不支持 XYZ 执行" : concurrencyError || undefined} onClick={start}>{submitBusy ? "正在提交…" : "开始 XYZ"}</button></div></footer>}
    {sync && <SyncDialog baseline={sync} draft={draft} onSync={onChange} onClose={() => setSync(null)} />}
  </section>;
}
