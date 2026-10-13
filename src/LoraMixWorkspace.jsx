import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ListOrdered, RefreshCw } from "lucide-react";
import XYZCanvasPanel from "./XYZCanvasPanel.jsx";
import { LoraMixRules, LoraMixSelection } from "./LoraMixMembers.jsx";
import { LoraMixBudget, LoraMixCandidates } from "./LoraMixPlan.jsx";
import { useDialogLifecycle } from "./gallery-core.js";
import { loraMixApi } from "./lora-mix-client.js";
import { loraMixRecipe } from "./lora-mix-draft.js";
import { generateCandidates } from "./lora-mix-generate.js";
import { mixIssues, mixRulesKey } from "./lora-mix-model.js";
import { vectorKey } from "./lora-mix-rules.js";
import { syncXYZBaseline } from "./xyz-draft.js";
import { xyzSidebarIssues } from "./xyz-sidebar-context.js";
import { xyzErrorText, xyzIssues, xyzSubmissionId, xyzSubmissionKey } from "./xyz-client.js";

const LoraMixResults = lazy(() => import("./LoraMixResults.jsx"));
const api = loraMixApi();
const syncLabels = { models: "模型 / 组件", prompts: "完整组合词 / 组快照", sampling: "采样 / 种子", canvas: "画布", loras: "全局 LoRA 完整有序列表 / 身份", postprocess: "固定后处理链", output: "输出命名" };
const errorPolicies = [{ value: "pause", label: "暂停并等待选择（默认）" }, { value: "continue", label: "记录失败并继续其余格" }];

function SyncDialog({ baseline, draft, onSync, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const [fields, setFields] = useState(Object.keys(syncLabels));
  return createPortal(<div className="xyz-sync-backdrop"><section className="xyz-sync-dialog mix-dialog panel" role="dialog" aria-modal="true" aria-labelledby="mix-sync-title" ref={ref}>
    <h2 id="mix-sync-title">从基础生图同步基线</h2>
    <p>仅覆盖勾选字段；X1 / X2 / X3、候选预览与执行设置不变。不会应用参数、卸载模型或修改真实 LoRA 组，也不会把搜索状态写入基础生图。</p>
    {Object.entries(syncLabels).map(([key, label]) => <label key={key} className="xyz-check"><input type="checkbox" checked={fields.includes(key)} onChange={(event) => setFields((old) => event.target.checked ? [...old, key] : old.filter((item) => item !== key))} />{label}</label>)}
    <p>采样：{draft.baseline.parameters.steps} → {baseline.parameters.steps} 步 · 画布：{baseline.parameters.width} × {baseline.parameters.height} · 全局挂载：{draft.baseline.loras.length} → {baseline.loras.length}</p>
    <div className="xyz-actions"><button type="button" onClick={onClose}>取消</button><button type="button" disabled={!fields.length} onClick={() => { onSync(syncXYZBaseline(draft, baseline, fields)); onClose(); }}>同步所选字段</button></div>
  </section></div>, document.body);
}

function IssueList({ issues, limit = 12 }) {
  return <ul className="xyz-issues">{issues.slice(0, limit).map((issue, index) => <li key={index}>{issue.message}</li>)}{issues.length > limit && <li>…另有 {issues.length - limit} 项</li>}</ul>;
}

export default function LoraMixWorkspace({ draft, onChange, captureBaseline, error, engine, focusExperiment, onFocusHandled, queueOpen, queueOutstanding, onToggleQueue, integration, promptDeck, canvasMaxEdge = 2048, loraManager, onNotice }) {
  const [sync, setSync] = useState(null), [tab, setTab] = useState("config"), [capabilities, setCapabilities] = useState(null), [onError, setOnError] = useState("pause");
  const [note, setNote] = useState(null), [used, setUsed] = useState(null), [rulesOpen, setRulesOpen] = useState(true);
  const [preview, setPreview] = useState(null), [previewKey, setPreviewKey] = useState(""), [planBusy, setPlanBusy] = useState(false), [planError, setPlanError] = useState(null);
  const [submitBusy, setSubmitBusy] = useState(false), [submitError, setSubmitError] = useState(null), [selected, setSelected] = useState(null), [refreshToken, setRefreshToken] = useState(0);
  const pendingSubmission = useRef(null), submitFlight = useRef(false), mounted = useRef(true), planRequest = useRef(0);
  const draftKey = useMemo(() => { if (!draft) return ""; try { return JSON.stringify(loraMixRecipe(draft)); } catch { return `invalid:${JSON.stringify(draft)}`; } }, [draft]);
  const draftKeyRef = useRef(draftKey), engineRef = useRef(engine || draft?.engine), lastDraftKey = useRef(draftKey);
  draftKeyRef.current = draftKey; engineRef.current = engine || draft?.engine;
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; planRequest.current += 1; }; }, []);
  useEffect(() => {
    let stopped = false;
    api.capabilities().then((value) => { if (!stopped) setCapabilities(value); }).catch(() => { if (!stopped) setCapabilities({ executor_ready: false, max_concurrency: 1 }); });
    return () => { stopped = true; };
  }, []);
  useEffect(() => {
    if (lastDraftKey.current === draftKey) return;
    lastDraftKey.current = draftKey;
    planRequest.current += 1; setPlanBusy(false); setPlanError(null); setSubmitError(null);
  }, [draftKey]);
  useEffect(() => { if (focusExperiment) { setTab("results"); setSelected({ experiment: focusExperiment }); onFocusHandled?.(); } }, [focusExperiment, onFocusHandled]);
  const report = useMemo(() => draft ? mixIssues(draft, capabilities?.limits ? { limits: { ...capabilities.limits } } : undefined) : null, [draft, capabilities]);
  // Tested / reserved vectors of the identical context, read from the server; they only shrink what a redraw may propose.
  const contextKey = draft ? JSON.stringify([mixRulesKey(draft), draft.baseline.loras.map((item) => [item.instanceId, item.weight, item.enabled]), draft.baseline.parameters.prompt, draft.baseline.parameters.seed, draft.repeat_count]) : "";
  useEffect(() => {
    if (!capabilities?.executor_ready || !draft || !report?.analysis.valid || !report.analysis.free.length) { setUsed(null); return undefined; }
    let stopped = false;
    const timer = window.setTimeout(() => {
      api.used(draft).then((value) => { if (!stopped) setUsed(value); }).catch(() => { if (!stopped) setUsed(null); });
    }, 400);
    return () => { stopped = true; window.clearTimeout(timer); };
  // The server resolves the context; re-ask whenever anything that defines it changes.
  }, [capabilities?.executor_ready, contextKey, refreshToken]); // eslint-disable-line react-hooks/exhaustive-deps
  const usedKeys = useMemo(() => {
    if (!used || !report) return new Set();
    return new Set((used.used || []).filter((entry) => entry.state === "tested" || entry.state === "reserved").map((entry) => {
      const units = report.analysis.ids.map((id) => entry.weights?.[id]);
      return units.every(Number.isInteger) ? vectorKey(units) : null;
    }).filter(Boolean));
  }, [used, report]);
  if (error) return <section className="xyz-workspace panel mix-workspace"><h2>LoRA 混合搜索草稿无法恢复</h2><p role="alert">{error}</p><p>原始草稿未删除；请先检查数据版本。不会自动从基础生图重建覆盖。</p></section>;
  if (!draft) return <section className="xyz-workspace panel mix-workspace" role="status">等待当前引擎基线就绪…</section>;
  const maxConcurrency = Math.max(1, Math.min(10, capabilities?.max_concurrency ?? 1));
  const baselineIssues = xyzSidebarIssues(draft);
  const blockingIssues = [...baselineIssues.map((message) => ({ message })), ...report.issues];
  const stale = Boolean(preview) && previewKey !== draftKey;
  const staleNote = draft.mix.candidates.length && draft.mix.proposed_key && draft.mix.proposed_key !== mixRulesKey(draft) ? "规则已在生成预览后变更：每一行已按新规则重新校验；建议重抽未固定项。" : "";
  const runPlan = async () => {
    if (planBusy || !report.valid || baselineIssues.length) return;
    const requestId = ++planRequest.current, submittedKey = draftKey, submittedEngine = engine || draft.engine, snapshot = structuredClone(draft);
    setPlanBusy(true); setPlanError(null);
    try {
      const value = await api.plan(snapshot);
      if (mounted.current && requestId === planRequest.current && draftKeyRef.current === submittedKey && engineRef.current === submittedEngine) { setPreview(value); setPreviewKey(submittedKey); }
    } catch (failure) {
      if (mounted.current && requestId === planRequest.current && draftKeyRef.current === submittedKey && engineRef.current === submittedEngine) setPlanError({ key: submittedKey, message: failure.message, issues: xyzIssues(failure.payload) });
    } finally { if (mounted.current && requestId === planRequest.current) setPlanBusy(false); }
  };
  const confirm = async () => {
    if (submitFlight.current || !capabilities?.executor_ready || !report.valid || baselineIssues.length) return;
    const submittedKey = draftKey, submittedEngine = engine || draft.engine, policy = onError, snapshot = structuredClone(draft);
    const submissionKey = xyzSubmissionKey(submittedKey, policy);
    if (!pendingSubmission.current || pendingSubmission.current.key !== submissionKey) pendingSubmission.current = { id: xyzSubmissionId(), key: submissionKey };
    const submissionId = pendingSubmission.current.id;
    submitFlight.current = true; setSubmitBusy(true); setSubmitError(null);
    try {
      const result = await api.confirm(snapshot, { submissionId, onError: policy });
      if (!mounted.current || engineRef.current !== submittedEngine) return;
      if (pendingSubmission.current?.id === submissionId) pendingSubmission.current = null;
      // The server owns the session: adopt its id/revision so the next confirmation extends the same lineage.
      if (draftKeyRef.current === submittedKey && result.session) onChange({ ...snapshot, mix: { ...snapshot.mix, session: { id: result.session.id, revision: result.session.revision } } });
      setSelected({ experiment: result.experiment.id, session: result.session?.id }); setRefreshToken((value) => value + 1); setTab("results");
      // No global toast here: it would sit over the run bar's primary action, and the results tab already shows the new round.
    } catch (failure) {
      if (!mounted.current || engineRef.current !== submittedEngine) return;
      if (failure.status && pendingSubmission.current?.id === submissionId) pendingSubmission.current = null;
      if (draftKeyRef.current === submittedKey) setSubmitError({ key: submittedKey, message: failure.message, issues: xyzIssues(failure.payload) });
    } finally { if (mounted.current && engineRef.current === submittedEngine) setSubmitBusy(false); submitFlight.current = false; }
  };
  // "Continue untested candidates": back to configuration with a fresh draw that excludes everything already tested or reserved.
  const continueSearch = () => {
    setTab("config");
    const result = generateCandidates(draft, { exclude: usedKeys });
    if (result.ok) { onChange(result.draft); setNote({ kind: result.note ? "info" : "success", message: result.note || `已按剩余空间生成 ${result.draft.mix.candidates.length} 个新候选预览（已排除已测试 / 已预留的向量）` }); }
    else setNote({ kind: "error", message: `无法继续：${result.issues.map((issue) => issue.message).join("；")}` });
  };
  // X1's rule chip: a collapsed X2 opens first, then the member's base field takes focus.
  const editRule = (index) => {
    setRulesOpen(true);
    window.requestAnimationFrame(() => {
      const field = document.getElementById(`mix-rule-${index}-base`);
      field?.scrollIntoView({ block: "center", behavior: "smooth" });
      field?.focus({ preventScroll: true });
    });
  };
  const canConfirm = capabilities?.executor_ready && report.valid && !baselineIssues.length && !submitBusy;
  const runState = planBusy ? "loading" : submitBusy ? "submitting" : blockingIssues.length || planError?.key === draftKey || submitError?.key === draftKey ? "error" : stale ? "stale" : preview?.ok ? "ready" : "unchecked";
  const total = report.expectedOutput;
  return <section className="xyz-workspace panel mix-workspace" aria-label="LoRA 混合搜索工作区">
    <div className="xyz-toolbar">
      <div className="xyz-tabs" role="group" aria-label="LoRA 混合搜索工作区视图"><button type="button" aria-pressed={tab === "config"} onClick={() => setTab("config")}>配置</button><button type="button" aria-pressed={tab === "results"} onClick={() => setTab("results")}>结果</button></div>
      <div className="xyz-head-actions">{onToggleQueue && <button type="button" className={queueOutstanding ? "busy" : ""} aria-expanded={Boolean(queueOpen)} title="查看任务队列" onClick={onToggleQueue}><ListOrdered size={14} />队列</button>}
        <button type="button" disabled={!captureBaseline} title="选择要同步的基础生图字段" onClick={() => setSync(captureBaseline())}><RefreshCw size={14} />同步基础基线</button></div>
    </div>
    {tab === "config" ? <div className="xyz-workspace-scroll mix-config">
      {promptDeck}
      {/* Same frame as XYZ: the canvas on the left and X1 beside it at the canvas card's height; X2 rules span below. */}
      <div className="xyz-canvas-axes mix-top">
        <XYZCanvasPanel draft={draft} onChange={onChange} canvasMaxEdge={canvasMaxEdge} />
        <LoraMixSelection draft={draft} manager={loraManager} analysis={report.analysis} issues={report.issues} onEditRule={editRule} onChange={onChange} />
      </div>
      <LoraMixRules draft={draft} analysis={report.analysis} issues={report.issues} open={rulesOpen} onToggle={setRulesOpen} onChange={onChange} />
      <LoraMixBudget draft={draft} report={report} capabilities={capabilities} policy={onError} policies={errorPolicies} onPolicy={setOnError} onChange={onChange} />
      <LoraMixCandidates draft={draft} report={report} used={usedKeys} usedCount={used ? usedKeys.size : null} staleNote={staleNote} note={note} onChange={onChange} onNote={setNote} />
      {preview && <section className={`xyz-plan-summary mix-plan-summary${preview.ok ? "" : " has-issues"}`} aria-live="polite"><h3>服务端预检{stale ? " · 草稿已修改，结果已过期" : ""}</h3>
        {preview.ok ? <p>{preview.summary?.profiles} 个配置 × {preview.summary?.samples} 样本 = {preview.summary?.cells} 张；新增 {preview.summary?.new ?? preview.summary?.profiles}，已测试/已预留 {preview.summary?.duplicates?.length ?? 0}。每格都已通过与普通生成相同的提交前检查；真实权重与显存仍在执行时复核。</p>
          : <><p role="alert">{preview.error_count ?? preview.errors?.length} 个问题需要先修正；提交会被拒绝且不预留向量：</p><IssueList issues={preview.errors || []} /></>}</section>}
      {planError?.key === draftKey && <div className="xyz-submit-error" role="alert"><p>预检失败 · {planError.message}</p>{planError.issues.length > 0 && <IssueList issues={planError.issues} />}</div>}
      {submitError?.key === draftKey && <div className="xyz-submit-error" role="alert"><p>确认失败 · {submitError.message}</p>{submitError.issues.length > 0 && <IssueList issues={submitError.issues} />}</div>}
      {blockingIssues.length > 0 && <details className="mix-issues" open><summary>需要修正的问题 · {blockingIssues.length}</summary><IssueList issues={blockingIssues} limit={20} /></details>}
    </div> : <Suspense fallback={<p role="status">正在加载结果…</p>}><LoraMixResults engine={engine || draft.engine} draft={draft} onChange={onChange} selected={selected} onSelect={setSelected} refreshToken={refreshToken} integration={integration} onNotice={onNotice} onContinue={continueSearch} /></Suspense>}
    {tab === "config" && <footer className="xyz-run-bar xyz-draft-run-bar mix-run-bar" data-state={runState} aria-label="当前搜索草稿运行操作">
      <div aria-live="polite"><strong className="xyz-run-total">{total === null || !draft.mix.candidates.length && report.analysis.free.length ? "待生成" : `${report.analysis.free.length ? total : draft.repeat_count} 张`}</strong>
        <span className="xyz-run-detail">{report.analysis.free.length ? `${draft.mix.candidates.length} 个配置 × ${draft.repeat_count} 样本` : `1 个配置 × ${draft.repeat_count} 样本`} · 预算 {report.budget.images ?? "?"} 张</span>
        <p className="xyz-run-status">{planBusy ? "正在服务端预检…" : submitBusy ? "正在确认回合…" : blockingIssues.length ? `${blockingIssues.length} 个问题待修正 · ${blockingIssues[0].message}` : stale ? "草稿已修改，预检过期 · 请重新预检" : !capabilities ? "正在读取执行服务能力…" : !capabilities.executor_ready ? "推理服务尚未提供混合搜索执行；草稿仍可编辑与保存。" : preview?.ok ? "预检通过 · 执行时仍校验真实权重与资源。" : "尚未预检 · 本地计数不是服务端准入。"}</p></div>
      <div className="xyz-run-actions"><button type="button" disabled={planBusy || !report.valid || baselineIssues.length > 0 || submitBusy || !capabilities?.executor_ready} title={capabilities && !capabilities.executor_ready ? "推理服务尚未提供混合搜索" : undefined} onClick={runPlan}>{planBusy ? "正在预检…" : "服务端预检"}</button>
        <button type="button" className="primary" disabled={!canConfirm} title={capabilities && !capabilities.executor_ready ? "推理服务尚未提供混合搜索执行" : undefined} onClick={confirm}>{submitBusy ? "正在确认…" : "确认并运行"}</button></div>
    </footer>}
    {sync && <SyncDialog baseline={sync} draft={draft} onSync={onChange} onClose={() => setSync(null)} />}
  </section>;
}
