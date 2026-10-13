import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, FileDown, Maximize2, MoreHorizontal, Star } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import TaskDeletionManager from "./TaskDeletionManager.jsx";
import { removedExperiments } from "./task-deletion.js";
import { ExportDialog } from "./XYZPlotWorkspace.jsx";
import { XYZCellApplyDialog } from "./XYZCellActions.jsx";
import { useDialogLifecycle } from "./gallery-core.js";
import { loraMixApi } from "./lora-mix-client.js";
import { GROUP_MEMBER_LIMIT, adoptAsBase, applyAdoptAsBase, followUp, groupFromCandidate, isFavorite, recommendedColumns, toggleFavorite } from "./lora-mix-followup.js";
import { formatUnits } from "./lora-mix-rules.js";
import { xyzBasicPrompt, xyzCellGalleryOutput, xyzCellSettings, xyzViewerSource } from "./xyz-apply.js";
import { XYZ_CELL_STATUS_TEXT, XYZ_EXPERIMENT_STATUS_TEXT, xyzApi, xyzExperimentActive } from "./xyz-client.js";

const mixApi = loraMixApi(), xyz = xyzApi();
const STATE_TEXT = { tested: "已测试", reserved: "已预留", failed: "未成功", cancelled: "已取消" };
const TILE_SIZES = [{ value: "132", label: "小" }, { value: "180", label: "中" }, { value: "260", label: "大" }];
const COLUMN_CHOICES = [2, 3, 4, 5, 6, 8, 10, 12];
const FOLLOW_UPS = [["adopt", "用作新基线…"], ["narrow", "更窄的后续…"], ["finer", "更细的后续…"], ["narrow+finer", "更窄且更细…"], ["group", "保存为组合…"]];
const text = (units) => formatUnits(units);

function Details({ round, candidate, cell, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const members = round.rules.members;
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog mix-dialog mix-details panel" role="dialog" aria-modal="true" aria-labelledby="mix-details-title" ref={ref}>
    <h2 id="mix-details-title">{candidate.id} · 配置详情</h2>
    <p>以下是服务端冻结的记录，不随当前草稿变化。总位移 {text(candidate.displacement)} · {candidate.source === "baseline" ? "基线（全部成员取基础权重）" : candidate.source === "manual" ? "手动配置" : "抽样配置"}。</p>
    <h3>测试 LoRA（按 X1 顺序）</h3>
    <table className="mix-detail-table"><thead><tr><th scope="col">LoRA</th><th scope="col">权重</th><th scope="col">规则</th></tr></thead><tbody>{members.map((member) => <tr key={member.ref.instanceId}>
      <td title={member.ref.value}>{member.ref.name || member.ref.value}</td><td className="mix-num">{text(candidate.weights[member.ref.instanceId])}</td>
      <td>{member.rule.locked ? "锁定 · 固定基础" : `范围 ${member.rule.min} … ${member.rule.max} · 步长 ${member.rule.step}`}</td></tr>)}</tbody></table>
    <h3>全局 LoRA（整轮固定背景）</h3>
    {round.global_loras.length ? <table className="mix-detail-table"><thead><tr><th scope="col">LoRA</th><th scope="col">权重</th><th scope="col">状态</th></tr></thead><tbody>{round.global_loras.map((item) => <tr key={item.instanceId || item.value}>
      <td title={item.value}>{item.name || item.value}</td><td className="mix-num">{String(item.weight)}</td><td>{item.enabled === false ? "停用" : "启用"}</td></tr>)}</tbody></table> : <p>无全局 LoRA。</p>}
    <p className="mix-keys"><code title="候选键 = 生成条件 + 规范权重">候选键 {candidate.key.slice(0, 16)}…</code> <code title="生成条件哈希">条件 {round.context_hash.slice(0, 16)}…</code>{cell?.seed ? <> · <code>种子 {cell.seed}</code></> : null}</p>
    <div className="xyz-actions"><button type="button" onClick={onClose}>关闭</button></div>
  </section></div>, document.body);
}

function FollowUpDialog({ draft, candidate, mode, onApply, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const plan = useMemo(() => mode === "adopt" ? { ...adoptAsBase(draft, candidate.weights), draft: null } : followUp(draft, candidate.weights, { narrow: mode.includes("narrow"), finer: mode.includes("finer") }), [draft, candidate, mode]);
  const title = { adopt: "用作新基线", narrow: "更窄的后续搜索", finer: "更细的后续搜索", "narrow+finer": "更窄且更细的后续搜索" }[mode];
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog mix-dialog panel" role="dialog" aria-modal="true" aria-labelledby="mix-follow-title" ref={ref}>
    <h2 id="mix-follow-title">{title} · {candidate.id}</h2>
    <p>只修改当前搜索草稿的 X2 规则：{mode === "adopt" ? "把基础权重设为该候选的权重。" : "基础权重移到该候选，范围 / 步长按下方预览调整。"}锁定成员、全局 LoRA、旧回合与真实组合都不会改变；候选预览会被清空。</p>
    {plan.ok ? <table className="mix-detail-table"><thead><tr><th scope="col">LoRA</th><th scope="col">基础</th>{mode !== "adopt" && <><th scope="col">最小</th><th scope="col">最大</th><th scope="col">步长</th></>}</tr></thead><tbody>
      {(mode === "adopt" ? plan.changes.map((item) => ({ name: item.name, base: `${item.from} → ${item.to}` })) : plan.changes.map((item) => ({ name: item.name, base: `${item.before.base} → ${item.after.base}`, min: `${item.before.min} → ${item.after.min}`, max: `${item.before.max} → ${item.after.max}`, step: `${item.before.step} → ${item.after.step}` }))).map((row) => <tr key={row.name}>
        <td>{row.name}</td><td className="mix-num">{row.base}</td>{mode !== "adopt" && <><td className="mix-num">{row.min}</td><td className="mix-num">{row.max}</td><td className="mix-num">{row.step}</td></>}</tr>)}</tbody></table>
      : <ul className="xyz-issues" role="alert">{plan.issues.map((message) => <li key={message}>{message}</li>)}</ul>}
    {plan.ok && plan.displacement && plan.displacement.before !== plan.displacement.after && <p className="mix-stale" role="status">可达位移上限已缩小：最大总位移 {plan.displacement.before} → {plan.displacement.after}（仅因旧值会被拒绝才调整，已在此明确显示）。</p>}
    <div className="xyz-actions"><button type="button" onClick={onClose}>取消</button><button type="button" className="primary" disabled={!plan.ok} onClick={() => { onApply(mode === "adopt" ? applyAdoptAsBase(draft, candidate.weights).draft : plan.draft); onClose(); }}>应用到搜索草稿</button></div>
  </section></div>, document.body);
}

function SaveGroupDialog({ draft, candidate, onSave, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const [scope, setScope] = useState("tests"), [message, setMessage] = useState("");
  const plan = groupFromCandidate(draft, candidate, { includeGlobals: scope === "complete" });
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog mix-dialog panel" role="dialog" aria-modal="true" aria-labelledby="mix-save-title" ref={ref}>
    <h2 id="mix-save-title">保存为 LoRA 组合 · {candidate.id}</h2>
    <p>始终新建一个组合，不会覆盖任何已有组合。请明确选择内容：</p>
    <label className="xyz-check"><input type="radio" name="mix-save-scope" checked={scope === "tests"} onChange={() => setScope("tests")} />仅测试成员（{draft.mix.members.length} 项，按该候选的权重）</label>
    <label className="xyz-check"><input type="radio" name="mix-save-scope" checked={scope === "complete"} onChange={() => setScope("complete")} />完整混合，包含当前草稿的全局 LoRA（{draft.baseline.loras.length + draft.mix.members.length} 项，上限 {GROUP_MEMBER_LIMIT}）</label>
    {plan.ok ? <p>将创建「{plan.name}」，共 {plan.members.length} 项{scope === "complete" ? "；全局背景与测试成员在组合中不再区分，提示词不写入组合" : ""}。</p> : <p role="alert">{plan.message}</p>}
    {message && <p role="alert">{message}</p>}
    <div className="xyz-actions"><button type="button" onClick={onClose}>取消</button><button type="button" className="primary" disabled={!plan.ok} onClick={() => { const problem = onSave(plan); if (problem) setMessage(problem); else onClose(); }}>创建组合</button></div>
  </section></div>, document.body);
}

/**
 * A tile's overflow menu. It is portalled with fixed coordinates: the grid scrolls on its own and the workspace clips,
 * so an in-flow popover would be cut off at the first column or the last row.
 */
function TileMenu({ id, items }) {
  const [place, setPlace] = useState(null);
  const trigger = useRef(null), menu = useRef(null);
  useEffect(() => {
    if (!place) return undefined;
    const outside = (event) => { if (!menu.current?.contains(event.target) && !trigger.current?.contains(event.target)) setPlace(null); };
    const dismiss = (event) => { if (!menu.current?.contains(event.target)) setPlace(null); };
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    menu.current?.querySelector("button:not(:disabled)")?.focus({ preventScroll: true });
    return () => { document.removeEventListener("pointerdown", outside, true); document.removeEventListener("scroll", dismiss, true); window.removeEventListener("resize", dismiss); };
  }, [place]);
  const open = () => {
    if (place) { setPlace(null); return; }
    const rect = trigger.current.getBoundingClientRect(), width = 188;
    const height = items.reduce((total, item) => total + (item.separator ? 9 : 32), 10);
    const below = window.innerHeight - rect.bottom - 8 >= height || rect.top < height + 8;
    setPlace({ width, left: Math.min(Math.max(8, rect.right - width), window.innerWidth - width - 8), top: below ? Math.min(rect.bottom + 4, window.innerHeight - height - 8) : rect.top - height - 4 });
  };
  const close = () => { trigger.current?.focus({ preventScroll: true }); setPlace(null); };
  const keyDown = (event) => {
    const buttons = [...menu.current.querySelectorAll("button:not(:disabled)")], at = buttons.indexOf(document.activeElement);
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); }
    else if (event.key === "Tab") setPlace(null);
    else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) && buttons.length) {
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (at + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].focus();
    }
  };
  return <>
    <button ref={trigger} type="button" className="mix-more-trigger" aria-label={`${id} 更多`} title="更多操作" aria-haspopup="menu" aria-expanded={Boolean(place)} onClick={open}><MoreHorizontal size={14} /></button>
    {place && createPortal(<div ref={menu} className="mix-more-menu" role="menu" aria-label={`${id} 更多操作`} style={place} onKeyDown={keyDown}>
      {items.map((item) => item.separator ? <hr key={item.key} /> : <button type="button" role="menuitem" key={item.key} disabled={item.disabled} title={item.title} onClick={() => { close(); item.onSelect(); }}>{item.label}</button>)}
    </div>, document.body)}
  </>;
}

export default function LoraMixResults({ engine, draft, onChange, selected, onSelect, refreshToken, integration = {}, onContinue }) {
  const [sessions, setSessions] = useState(null), [view, setView] = useState(null), [detail, setDetail] = useState(null), [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [sample, setSample] = useState(0), [columns, setColumns] = useState("auto"), [size, setSize] = useState("180"), [favoritesOnly, setFavoritesOnly] = useState(false);
  const [dialog, setDialog] = useState(null), [exporting, setExporting] = useState(false), [exportLoras, setExportLoras] = useState(false), [applyCell, setApplyCell] = useState(""), [galleryBusy, setGalleryBusy] = useState("");
  // Result actions report inline: a global toast would sit over the run bar's primary action.
  const [status, setStatus] = useState("");
  const request = useRef(0), listRequest = useRef(0), mounted = useRef(true), engineRef = useRef(engine), selectedRef = useRef(selected), deletedIds = useRef(new Set());
  engineRef.current = engine; selectedRef.current = selected;
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; request.current += 1; }; }, []);
  const loadSessions = useCallback(async () => {
    const token = ++listRequest.current;
    try { const value = await mixApi.sessions(engine); if (mounted.current && engineRef.current === engine && token === listRequest.current) { setSessions(value.sessions); return value.sessions; } } catch (failure) { if (mounted.current && engineRef.current === engine && token === listRequest.current) setError(failure.message); }
  }, [engine]);
  useEffect(() => { request.current++; listRequest.current++; setSessions(null); setDetail(null); setView(null); setDialog(null); setExporting(false); setApplyCell(""); setBusy(false); setGalleryBusy(""); setError(""); return () => { request.current++; listRequest.current++; }; }, [engine]);
  useEffect(() => { void loadSessions(); }, [loadSessions, refreshToken]);
  const proposedId = selected?.experiment || draft.mix.session.id && view?.id === draft.mix.session.id && view.rounds.at(-1)?.experiment_id || sessions?.[0]?.latest_experiment || null;
  const experimentId = deletedIds.current.has(proposedId) ? sessions?.find((item) => !deletedIds.current.has(item.latest_experiment))?.latest_experiment || null : proposedId;
  const experimentRef = useRef(experimentId); experimentRef.current = experimentId;
  const load = useCallback(async (id) => {
    const token = ++request.current;
    if (!id) { setDetail(null); setView(null); return null; }
    try {
      const value = await xyz.detail(id);
      const sessionId = value.origin?.session_id;
      const session = sessionId ? await mixApi.session(sessionId) : null;
      if (!mounted.current || token !== request.current || experimentRef.current !== id || deletedIds.current.has(id)) return null;
      setDetail(value); setView(session); setError("");
      return value;
    } catch (failure) { if (mounted.current && token === request.current && experimentRef.current === id) setError(failure.message); return null; }
  }, []);
  useEffect(() => {
    let stopped = false, timer = 0;
    request.current++; setDetail(null); setView(null); setDialog(null); setExporting(false); setApplyCell(""); setBusy(false); setGalleryBusy(""); setSample(0); setStatus("");
    const tick = async () => {
      const value = await load(experimentId);
      if (!stopped && xyzExperimentActive(value)) timer = window.setTimeout(tick, 1500); else if (!stopped) void loadSessions();
    };
    void tick();
    return () => { stopped = true; request.current++; window.clearTimeout(timer); };
  }, [experimentId, load, loadSessions, refreshToken]);
  const origin = detail?.origin, known = origin?.known !== false && origin?.kind === "lora_mix_search";
  const round = known && view ? view.rounds.find((item) => item.experiment_id === detail.id) : null;
  const candidates = round?.candidates || [];
  const samples = detail?.repeat_count || 1;
  const effectiveColumns = columns === "auto" ? recommendedColumns(candidates.length) : Number(columns);
  const act = async (run, success) => {
    if (busy) return;
    const id = experimentId;
    setBusy(true);
    try { await run(); if (!mounted.current || experimentRef.current !== id || deletedIds.current.has(id)) return; await load(id); await loadSessions(); if (success && experimentRef.current === id) setStatus(success); } catch (failure) { if (mounted.current && experimentRef.current === id) setError(failure.message); } finally { if (mounted.current && experimentRef.current === id) setBusy(false); }
  };
  const actions = useMemo(() => ({
    open: integration.onOpenViewer ? (cell) => integration.onOpenViewer(xyzViewerSource({ assetId: cell.asset_id, name: cell.output_name })) : null,
    gallery: integration.onAddToGallery && detail ? async (cell) => {
      const experiment = detail.id;
      setGalleryBusy(cell.id);
      try {
        const full = await xyz.cell(detail.id, cell.id), context = integration.promptContext || {};
        if (!mounted.current || experimentRef.current !== experiment || deletedIds.current.has(experiment)) return;
        const prompt = xyzBasicPrompt(full.prompt_parts, { ...context, groupPrompt: context.groupPromptFor ? context.groupPromptFor(null) : "" });
        integration.onAddToGallery({ outputs: [xyzCellGalleryOutput({ ...full, thumbnail_url: cell.thumbnail_url })], settings: xyzCellSettings(full, prompt) });
        if (mounted.current) setStatus(`已把 ${cell.labels?.x || cell.id} 加入画廊（使用本格实际参数）`);
      } catch (failure) { if (mounted.current && experimentRef.current === experiment) setError(failure.message); } finally { if (mounted.current && experimentRef.current === experiment) setGalleryBusy(""); }
    } : null,
  }), [integration, detail]);
  const deleted = async (result) => {
    const ids = removedExperiments(result);
    ids.forEach((id) => deletedIds.current.add(id));
    const removedSessions = result.sessionVisibilityRemoved || result.results.filter((entry) => entry.outcome === "deleted").map((entry) => entry.id);
    setSessions((old) => old?.filter((item) => !removedSessions.includes(item.id)) ?? old);
    if (ids.includes(experimentRef.current) || ids.includes(selectedRef.current?.experiment)) {
      request.current++; setDetail(null); setView(null); setDialog(null); setExporting(false); setApplyCell(""); setGalleryBusy(""); onSelect(null);
    }
    await loadSessions();
  };
  const missingDraftSession = draft.mix.session.id && sessions && !sessions.some((item) => item.id === draft.mix.session.id);
  const recovery = missingDraftSession ? <div className="xyz-note task-delete-recovery"><p>当前草稿关联的搜索任务已不在结果列表。不会自动重建旧会话；规则、预算与候选历史均保留。</p><button type="button" onClick={() => onChange({ ...draft, mix: { ...draft.mix, session: { id: null, revision: 0 } } })}>以当前草稿新建搜索</button></div> : null;
  const manager = <TaskDeletionManager api={mixApi} module="lora_mix" engine={engine} refreshToken={refreshToken} onDeleted={deleted} currentId={view?.id || null} />;
  // The header row (pickers + task manager) is the first child in every state, so the manager never remounts mid-deletion.
  const head = (picks) => <div className="mix-results-head"><div className="mix-results-picks">{picks}</div>{manager}</div>;
  const shell = (body) => <section className="xyz-results mix-results" aria-label="LoRA 混合搜索结果"><div className="mix-results-main">{head(null)}{recovery}{error && <p role="alert" className="xyz-error">{error}</p>}{body}</div></section>;
  if (!sessions && !detail && !error) return shell(<p role="status" className="xyz-empty-results">正在读取搜索记录…</p>);
  if (!experimentId) return shell(<p className="xyz-empty-results" role="status">此引擎还没有确认过搜索回合。确认并运行后，结果会按候选配置显示在这里。</p>);
  if (detail && !known) return shell(<p role="alert">此实验的来源（{origin?.kind || "未知"}）不是本版本可解读的搜索回合；原始记录已保留，不会被当作搜索结果打开。</p>);
  const active = xyzExperimentActive(detail), counts = detail?.counts, total = detail?.total_cells || 0;
  const finished = counts ? counts.complete + counts.error + counts.skipped + counts.cancelled + counts.missing : 0;
  const unfinished = counts ? counts.pending + counts.cancelled : 0, failed = counts ? counts.error + counts.skipped + counts.missing : 0;
  const visible = candidates.filter((candidate) => !favoritesOnly || isFavorite(draft, candidate.key));
  const rounds = view?.rounds || [];
  const findCandidate = (id) => candidates.find((candidate) => candidate.id === id);
  const cellOf = (candidate) => detail.cells.find((item) => item.coordinate[0] === candidate.x_index && item.coordinate[3] === sample);
  const aspect = detail?.baseline?.width > 0 && detail?.baseline?.height > 0 ? `${detail.baseline.width} / ${detail.baseline.height}` : "1 / 1";
  const sessionOptions = (sessions || []).map((item) => ({ value: item.id, label: `${new Date(item.updated_at * 1000).toLocaleString()} · ${item.rounds} 轮 · ${item.id.slice(0, 8)}` }));
  if (view && !sessionOptions.some((option) => option.value === view.id)) sessionOptions.unshift({ value: view.id, label: `${view.rounds.length} 轮 · ${view.id.slice(0, 8)}` });
  const roundOptions = rounds.map((item) => ({ value: item.experiment_id, label: `第 ${item.round} 轮 · ${XYZ_EXPERIMENT_STATUS_TEXT[item.status] || item.status} · ${item.candidates.filter((c) => c.state === "tested").length}/${item.candidates.length} 已测试` }));
  const menuItems = (candidate, cell) => {
    const own = [
      ...(actions.gallery ? [{ key: "gallery", label: galleryBusy && galleryBusy === cell?.id ? "正在加入画廊…" : "加入画廊", title: "使用本格实际参数加入画廊", disabled: !cell?.asset_url || galleryBusy === cell.id, onSelect: () => actions.gallery(cell) }] : []),
      ...(integration.onApplyToBasic ? [{ key: "apply", label: "应用到基础生图…", title: "逐组选择要应用的参数", disabled: !cell, onSelect: () => setApplyCell(cell.id) }] : []),
    ];
    return [...own, ...(own.length ? [{ key: "separator", separator: true }] : []), ...FOLLOW_UPS.map(([kind, label]) => ({ key: kind, label, onSelect: () => setDialog({ kind, id: candidate.id }) }))];
  };
  return <section className="xyz-results mix-results" aria-label="LoRA 混合搜索结果">
    <div className="mix-results-main">
      {head(<>
        <div className="mix-pick"><span className="mix-label">搜索会话</span><WorkspaceSelect ariaLabel="选择搜索会话" value={view?.id || ""} options={sessionOptions} placeholder="选择会话"
          onChange={(id) => { const next = sessions?.find((item) => item.id === id); if (next) onSelect({ experiment: next.latest_experiment, session: next.id }); }} /></div>
        <div className="mix-pick"><span className="mix-label">回合</span><WorkspaceSelect ariaLabel="选择搜索回合" value={detail?.id || ""} options={roundOptions} placeholder="选择回合" onChange={(id) => onSelect({ experiment: id, session: view?.id })} /></div>
      </>)}
      {recovery}
      {error && <p role="alert" className="xyz-error">{error}</p>}
      {status && <p className="mix-note success" role="status">{status}</p>}
      {detail && round && <>
        <section className="mix-round" aria-label="回合状态">
          <header className="mix-round-head"><strong>{XYZ_EXPERIMENT_STATUS_TEXT[detail.status] || detail.status}</strong><span>{counts.complete}/{total} 格完成{failed ? ` · ${failed} 未成功` : ""}</span>
            <small>全局 LoRA {round.global_loras.length} 个 · 测试 LoRA {round.rules.members.length} 个 · 种子 {round.seeds.join("、")} · 条件 {round.context_hash.slice(0, 8)}</small></header>
          <div className="xyz-progress" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={finished}><i style={{ width: `${total ? Math.round(100 * finished / total) : 0}%` }} /></div>
          <p className="mix-remaining"><FileDown size={13} aria-hidden="true" />尚未测试的合法向量（按该回合规则，排除同一条件下已测试 / 已预留的）：<a href={`/api/inference/lora-mix/sessions/${view.id}/remaining?format=csv&round=${round.round}&limit=1000`} download={`lora-mix-${view.id.slice(0, 8)}-round${round.round}-remaining.csv`}>CSV（前 1000 项）</a> · <a href={`/api/inference/lora-mix/sessions/${view.id}/remaining?format=json&round=${round.round}&limit=200`} target="_blank" rel="noreferrer">JSON 分页</a>。大空间按页惰性枚举，不会一次列出全部。</p>
        </section>
        {detail.decision && <div className="xyz-decision" role="alert"><p>格 {detail.decision.cell_id} 失败：{detail.decision.error}</p><div className="xyz-actions">
          <button type="button" disabled={busy} onClick={() => act(() => xyz.decide(detail.id, "retry"))}>重试此格（同一冻结请求与种子）</button>
          <button type="button" disabled={busy} onClick={() => act(() => xyz.decide(detail.id, "skip"))}>跳过此格并继续</button>
          <button type="button" disabled={busy} onClick={() => act(() => xyz.decide(detail.id, "stop"))}>停止回合</button></div></div>}
        <div className="mix-grid-tools" role="group" aria-label="候选网格显示">
          <div className="mix-pick"><span className="mix-label" title="仅显示，不是轴">每行列数</span><WorkspaceSelect ariaLabel="候选网格列数" value={columns} onChange={setColumns}
            options={[{ value: "auto", label: `推荐（${recommendedColumns(candidates.length)} 列）` }, ...COLUMN_CHOICES.map((value) => ({ value: String(value), label: `${value} 列` }))]} /></div>
          <div className="mix-pick"><span className="mix-label" title="不影响导出">缩略图</span><WorkspaceSelect ariaLabel="候选缩略图大小" value={size} options={TILE_SIZES} onChange={setSize} /></div>
          {samples > 1 && <div className="mix-pick"><span className="mix-label">样本</span><WorkspaceSelect ariaLabel="选择重复样本" value={sample} onChange={setSample}
            options={Array.from({ length: samples }, (_, index) => ({ value: index, label: `样本 ${index + 1} · 种子 ${round.seeds[index]}` }))} /></div>}
          <label className="xyz-check mix-favorites"><input type="checkbox" checked={favoritesOnly} onChange={(event) => setFavoritesOnly(event.target.checked)} />只看收藏</label>
          <button type="button" className="mix-export-entry" disabled={!counts.complete} title={counts.complete ? "导出整个候选网格为单张 PNG" : "至少需要 1 张已完成原图"} onClick={() => setExporting(true)}><Download size={14} />导出候选网格 PNG…</button>
        </div>
        <div className="mix-grid" role="list" aria-label={`第 ${round.round} 轮候选网格`} style={{ "--mix-columns": effectiveColumns, "--mix-tile": `${size}px`, "--mix-aspect": aspect }}>
          {visible.map((candidate) => {
            const cell = cellOf(candidate), favorite = isFavorite(draft, candidate.key);
            return <figure key={candidate.id} role="listitem" className={`mix-tile ${cell?.status || "pending"}${favorite ? " favorite" : ""}`}>
              <div className="mix-tile-image">{cell?.status === "complete" && cell.asset_url
                ? actions.open && cell.asset_id ? <button type="button" className="xyz-cell-open" aria-label={`预览 ${candidate.id} 原图`} onClick={() => actions.open(cell)}><img src={cell.thumbnail_url || cell.asset_url} alt={`${candidate.id} 原图`} loading="lazy" decoding="async" /></button>
                  : <a href={cell.asset_url} target="_blank" rel="noreferrer" aria-label={`打开 ${candidate.id} 原图`}><img src={cell.thumbnail_url || cell.asset_url} alt={`${candidate.id} 原图`} loading="lazy" decoding="async" /></a>
                : <span className="xyz-cell-placeholder">{XYZ_CELL_STATUS_TEXT[cell?.status] || "等待"}{cell?.attempts > 1 ? ` · 第 ${cell.attempts} 次` : ""}</span>}</div>
              <figcaption>
                <span className="mix-id">{candidate.id}</span>{candidate.source === "baseline" && <span className="mix-badge" title="所有成员取基础权重">基线</span>}
                <span className="mix-num" title="相对固定基础权重的总位移">Δ{text(candidate.displacement)}</span>
                <span className={`mix-state ${candidate.state}`}>{STATE_TEXT[candidate.state] || candidate.state}{samples > 1 ? ` ${candidate.complete}/${candidate.samples}` : ""}</span>
              </figcaption>
              <div className="mix-tile-actions" role="group" aria-label={`${candidate.id} 操作`}>
                <button type="button" className="mix-favorite" aria-pressed={favorite} aria-label={`${favorite ? "取消收藏" : "收藏"} ${candidate.id}`} title={favorite ? "取消收藏" : "收藏"} onClick={() => onChange(toggleFavorite(draft, candidate.key))}><Star size={13} /></button>
                <button type="button" className="mix-text-button" title="冻结权重与全局背景" onClick={() => setDialog({ kind: "details", id: candidate.id })}>详情</button>
                {cell?.asset_url && actions.open && cell.asset_id && <button type="button" aria-label={`预览 ${candidate.id}`} title="在预览工作区打开原图" onClick={() => actions.open(cell)}><Maximize2 size={13} /></button>}
                {cell?.asset_url && <a className="xyz-cell-download" href={cell.asset_url} download={cell.output_name || true} aria-label={`下载 ${candidate.id} 原图`} title="下载原图"><Download size={13} /></a>}
                <TileMenu id={candidate.id} items={menuItems(candidate, cell)} />
              </div>
            </figure>;
          })}
          {!visible.length && <p className="mix-empty" role="status">{favoritesOnly ? "没有收藏的候选。" : "这一轮没有候选。"}</p>}
        </div>
      </>}
    </div>
    {detail && round && <footer className="xyz-run-bar xyz-result-run-bar mix-result-bar" aria-label="选中回合运行操作">
      <div><strong>第 {round.round} 轮 · {XYZ_EXPERIMENT_STATUS_TEXT[detail.status] || detail.status}</strong><p>{counts.complete}/{total} 格 · 请求并发 {detail.requested_concurrency} / 实际 {detail.effective_concurrency ?? "—"}{detail.foreign ? " · 其他进程拥有，只读" : ""}</p></div>
      <div className="xyz-run-actions">
        {active && !detail.decision && ["queued", "running"].includes(detail.status) && <button type="button" disabled={busy || detail.foreign} onClick={() => act(() => xyz.control(detail.id, "pause"))}>暂停</button>}
        {active && !detail.decision && ["pausing", "paused"].includes(detail.status) && <button type="button" disabled={busy || detail.foreign} onClick={() => act(() => xyz.control(detail.id, "resume"))}>{detail.status === "paused" ? "继续生成" : "取消暂停并继续"}</button>}
        {active && !detail.decision && detail.status !== "cancelling" && <button type="button" disabled={busy || detail.foreign} onClick={() => act(() => xyz.control(detail.id, "stop"))}>停止（保留已完成原图）</button>}
        {!active && !detail.foreign && unfinished > 0 && <button type="button" disabled={busy} onClick={() => act(() => xyz.continue(detail.id, "none"), "已继续未完成的格")}>继续未完成的 {unfinished} 格</button>}
        {!active && !detail.foreign && failed > 0 && <button type="button" disabled={busy} onClick={() => act(() => xyz.continue(detail.id, "failed"), "已重试失败的格（生成新文件）")}>重试失败 / 缺失的 {failed} 格</button>}
        {["interrupted", "cancelled", "error"].includes(detail.status) && !detail.foreign && unfinished + failed > 0 && !detail.mix_released && <button type="button" disabled={busy} title="只释放未成功测试的向量，让后续轮次可以再次选中；已完成原图不受影响" onClick={() => act(() => mixApi.release(detail.id), "已释放未执行的向量预留")}>释放未执行预留</button>}
        <button type="button" className="primary" disabled={busy || active} title={active ? "回合执行中；完成后可继续" : "回到配置，按剩余空间抽取下一轮（不会再次选中已测试 / 已预留向量）"} onClick={() => onContinue?.()}>继续未测试候选</button>
      </div>
    </footer>}
    {dialog?.kind === "details" && round && findCandidate(dialog.id) && <Details round={round} candidate={findCandidate(dialog.id)} cell={cellOf(findCandidate(dialog.id))} onClose={() => setDialog(null)} />}
    {["adopt", "narrow", "finer", "narrow+finer"].includes(dialog?.kind) && findCandidate(dialog.id) && <FollowUpDialog draft={draft} candidate={findCandidate(dialog.id)} mode={dialog.kind} onApply={(next) => { onChange(next); setStatus("已更新搜索草稿的 X2 规则；请回到配置重新生成候选"); }} onClose={() => setDialog(null)} />}
    {dialog?.kind === "group" && findCandidate(dialog.id) && <SaveGroupDialog draft={draft} candidate={findCandidate(dialog.id)} onSave={(plan) => { const problem = integration.onSaveGroup ? integration.onSaveGroup(plan) : "当前页面无法保存组合"; if (!problem) setStatus(`已新建组合「${plan.name}」（${plan.members.length} 项）；没有修改任何已有组合`); return problem; }} onClose={() => setDialog(null)} />}
    {exporting && detail && <ExportDialog detail={{ ...detail, repeat_count: samples }} page={{ z: 0, sample }} exportApi={mixApi.exports} extraBody={{ columns: columns === "auto" ? effectiveColumns : Number(columns), lora_weights: exportLoras }} blockLabel="样本页排列" captionLabel="每格下方显示编号 / 位移 / 状态（同一样本的种子写在摘要里）"
      extraControls={<label className="xyz-check"><input type="checkbox" checked={exportLoras} onChange={(event) => setExportLoras(event.target.checked)} />显示 LoRA 与权重（摘要下列出编号的测试 LoRA，每格下方标注对应编号的权重）</label>} onClose={() => setExporting(false)} onFinished={() => void load(experimentId)} onOpenViewer={integration.onOpenViewer} />}
    {applyCell && detail && <XYZCellApplyDialog experimentId={detail.id} cellId={applyCell} promptContext={integration.promptContext} onApply={integration.onApplyToBasic} onGoBasic={integration.onGoBasic} onClose={() => setApplyCell("")} />}
  </section>;
}
