import { useState } from "react";
import { Dices, Pin, PinOff, Plus, RefreshCw, Trash2, X } from "lucide-react";
import { XYZCountField } from "./SamplingControls.jsx";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import { BUDGET_PRESETS } from "./lora-mix-draft.js";
import { addManualCandidate, clearCandidates, drawSamplingSeed, editCandidateValue, generateCandidates, removeCandidate, togglePin } from "./lora-mix-generate.js";
import { formatUnits } from "./lora-mix-rules.js";
import { validSamplingSeed } from "./lora-mix-sampler.js";

const PAGE = 24;
const group = (text) => text.replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** Exact cardinalities as text: BigInt digits, never a lossy Number. */
export function spaceText(space) {
  if (!space) return "待规则与位移有效后计算";
  if (space.kind === "exact") return `${group(space.count.toString())} 个${space.constrained ? "（受位移约束，精确）" : "（未受位移约束，精确）"}`;
  return `未知（${space.reason || "过大"}；不会物化全部向量）`;
}

/** X3: bounded displacement, a round budget, the shared seed set and the run policy; nothing here starts a job. */
export function LoraMixBudget({ draft, report, capabilities, policy, policies, onPolicy, onChange }) {
  const mix = draft.mix, { analysis, budget } = report;
  const patch = (value) => onChange({ ...draft, mix: { ...mix, ...value } });
  const displacementError = report.issues.find((issue) => issue.scope === "displacement");
  const maxConcurrency = Math.max(1, Math.min(10, capabilities?.max_concurrency ?? 1));
  const seedValid = validSamplingSeed(mix.sampling_seed);
  return <section className="mix-x3 mix-card" aria-label="X3 位移与预算">
    <div className="section-heading"><span>X3</span><h2>位移与预算</h2><small className="mix-heading-meta">{budget.errors.length ? "预算待修正" : `预算 ${budget.images} 张`}</small></div>
    <div className="mix-x3-grid">
      <div className={`mix-field${displacementError ? " has-error" : ""}`}>
        <label htmlFor="mix-max-displacement">最大总位移</label>
        <div className="mix-inline"><input id="mix-max-displacement" type="text" inputMode="decimal" value={mix.max_displacement} aria-invalid={Boolean(displacementError) || undefined} aria-describedby={displacementError ? "mix-max-displacement-error" : "mix-max-displacement-hint"} disabled={!analysis.valid || !analysis.free.length} onChange={(event) => patch({ max_displacement: event.target.value })} />
          <button type="button" disabled={!analysis.valid || !analysis.free.length} title="明确设为可达上限（不是自动截断）" onClick={() => patch({ max_displacement: formatUnits(analysis.dmax) })}>完整范围</button></div>
        <small id="mix-max-displacement-hint">相对固定基础权重的 Σ|权重 − 基础|；仅计算未锁定测试成员，全局 LoRA 不参与。{analysis.valid && analysis.free.length ? `可达上限 ${formatUnits(analysis.dmax)}。` : analysis.valid ? "全部成员已锁定，只有 1 个配置。" : ""}</small>
        {displacementError && <small role="alert" id="mix-max-displacement-error">{displacementError.message}</small>}
      </div>
      <fieldset className="mix-budget"><legend>本轮图片预算</legend>
        <div className="mix-budget-row"><div className="mix-budget-options" role="radiogroup" aria-label="本轮图片预算">{[...BUDGET_PRESETS, "custom"].map((value) => <label key={value} className="mix-radio"><input type="radio" name="mix-budget" value={value} checked={mix.budget.preset === value} onChange={() => patch({ budget: { ...mix.budget, preset: value } })} /><span>{value === "custom" ? "自定义" : value}</span></label>)}</div>
          {mix.budget.preset === "custom" && <input type="text" inputMode="numeric" className="mix-budget-custom" aria-label="自定义本轮图片预算" value={mix.budget.custom} aria-invalid={budget.errors.length > 0 || undefined} onChange={(event) => patch({ budget: { ...mix.budget, custom: event.target.value } })} />}</div>
        <small>上限含基线 / 对照图与所有样本，不会被并发放大。{budget.errors.length ? "" : `共 ${budget.images} 张 = ${budget.slots} 个配置 × ${budget.seeds} 样本${budget.leftover ? `（${budget.leftover} 张余量不使用）` : ""}。`}</small>
        {budget.errors.map((message, index) => <small role="alert" key={index}>{message}</small>)}
      </fieldset>
      <div className="mix-counts xyz-counts xyz-mother-controls">
        <XYZCountField label="重复样本" detail="同一组图像种子用于所有配置；计入预算" value={draft.repeat_count} max={32} invalid={!Number.isInteger(draft.repeat_count) || draft.repeat_count < 1 || draft.repeat_count > 32} onChange={(repeat_count) => onChange({ ...draft, repeat_count })} />
        <XYZCountField label="请求并发" detail={`服务端上限 ${capabilities?.max_concurrency ?? "读取中"}；默认 1 顺序生成`} value={draft.requested_concurrency} max={maxConcurrency} invalid={!Number.isInteger(draft.requested_concurrency) || draft.requested_concurrency < 1 || draft.requested_concurrency > maxConcurrency} onChange={(requested_concurrency) => onChange({ ...draft, requested_concurrency })} />
      </div>
      <div className={`mix-field${seedValid ? "" : " has-error"}`}>
        <label htmlFor="mix-sampling-seed">采样种子（只决定候选抽样，与图像种子无关）</label>
        <div className="mix-inline"><input id="mix-sampling-seed" type="text" inputMode="numeric" value={mix.sampling_seed} aria-invalid={!seedValid || undefined} onChange={(event) => patch({ sampling_seed: event.target.value })} />
          <button type="button" title="明确抽取一个新的采样种子" onClick={() => patch({ sampling_seed: drawSamplingSeed() })}><Dices size={14} />重新抽取</button></div>
        {!seedValid && <small role="alert">采样种子必须是 uint64 十进制文本</small>}
      </div>
      {policies && <div className="mix-field">
        <span className="mix-label" id="mix-policy-label">单格失败策略</span>
        <WorkspaceSelect ariaLabel="单格失败策略" value={policy} options={policies} onChange={onPolicy} />
      </div>}
      <label className="mix-check"><input type="checkbox" checked={mix.include_baseline} onChange={(event) => patch({ include_baseline: event.target.checked })} />包含基线（所有成员取基础权重；占用 1 个配置名额）</label>
    </div>
    <p className="mix-summary" role="status" aria-live="polite">候选空间：{spaceText(report.space)}。{report.expectedOutput !== null ? `当前预览 ${mix.candidates.length} 个配置 → 预期输出 ${report.expectedOutput} 张。` : ""}</p>
  </section>;
}

/** The editable preview: raw text per cell, verdicts per row, pinning and redraw of the remainder. */
export function LoraMixCandidates({ draft, report, used, usedCount = null, staleNote, note, onChange, onNote }) {
  const [page, setPage] = useState(0);
  const { analysis } = report, mix = draft.mix, rows = report.candidates, pages = Math.max(1, Math.ceil(rows.length / PAGE)), current = Math.min(page, pages - 1);
  const labelOf = (id) => mix.members.find((member) => member.ref.instanceId === id)?.ref;
  const run = () => {
    const result = generateCandidates(draft, { exclude: used || new Set() });
    if (!result.ok) { onNote({ kind: "error", message: `无法生成候选：${result.issues.map((issue) => issue.message).join("；")}` }); return; }
    onChange(result.draft);
    onNote({ kind: result.note ? "info" : "success", message: result.note || `已生成 ${result.draft.mix.candidates.length} 个候选预览（仅在内存与草稿中；不占用向量、不提交任务）` });
    setPage(0);
  };
  const manual = () => { const result = addManualCandidate(draft); if (result.ok) { onChange(result.draft); setPage(Math.max(0, Math.ceil(result.draft.mix.candidates.length / PAGE) - 1)); } else onNote({ kind: "error", message: "先修正 X2 规则后再添加手动候选" }); };
  const ready = analysis.valid && analysis.free.length > 0;
  const invalid = rows.filter((entry) => entry.problems.length > 0).length;
  return <section className="mix-candidates mix-card" aria-label="候选预览">
    <div className="section-heading"><span>预览</span><h2>候选配置</h2><small className="mix-heading-meta">{rows.length ? `${rows.length} 行${invalid ? ` · ${invalid} 行待修正` : ""}` : "未生成"}</small></div>
    <div className="mix-candidate-actions" role="group" aria-label="候选预览操作">
      <button type="button" className="primary" disabled={!analysis.valid} onClick={run}><RefreshCw size={14} />{rows.length ? "重抽未固定项" : "生成候选预览"}</button>
      <button type="button" disabled={!ready} onClick={manual}><Plus size={14} />添加手动候选</button>
      <button type="button" disabled={!rows.length} onClick={() => { onChange(clearCandidates(draft)); onNote(null); }}><X size={14} />清空预览</button>
    </div>
    <p className="mix-hint">预览只存在于草稿：不预留向量、不创建任何任务。可直接编辑权重、固定（保留）、删除；每次编辑都会重新校验序列、范围、锁定、位移、重复与预算。</p>
    {note && <p className={`mix-note ${note.kind}`} role={note.kind === "error" ? "alert" : "status"}>{note.message}</p>}
    {usedCount !== null && <p className="mix-hint" role="status">同一生成条件下服务端已记录 {usedCount} 个已测试 / 已预留向量；重抽不会再次选中它们。生成条件变化（全局 LoRA、模型、提示词、种子等）会开启新的条件，旧图只作历史参考。</p>}
    {staleNote && <p className="mix-stale" role="status">{staleNote}</p>}
    {!analysis.free.length && analysis.valid && <p className="mix-empty" role="status">全部成员已锁定：只有 1 个配置（固定基础权重），无需候选行。</p>}
    {rows.length > 0 && <>
      <div className="mix-candidate-scroll"><table className="mix-candidate-table">
        <thead><tr><th scope="col">编号</th>{analysis.ids.map((id) => <th scope="col" key={id} title={labelOf(id)?.value}>{labelOf(id)?.name || labelOf(id)?.value}</th>)}<th scope="col">总位移</th><th scope="col">状态</th><th scope="col"><span className="sr-only">操作</span></th></tr></thead>
        <tbody>{rows.slice(current * PAGE, (current + 1) * PAGE).map((entry) => {
          const candidate = entry.candidate, bad = entry.problems.length > 0;
          return <tr key={candidate.id} className={bad ? "has-error" : ""} data-source={candidate.source}>
            <th scope="row"><span className="mix-id">{candidate.id}</span>{candidate.source === "baseline" && <span className="mix-badge" title="所有成员取基础权重">基线</span>}{candidate.pinned && <span className="mix-badge pinned">已固定</span>}</th>
            {analysis.ids.map((id) => {
              const cell = entry.problems.find((problem) => problem.id === id);
              return <td key={id}><input type="text" inputMode="decimal" aria-label={`${candidate.id} ${labelOf(id)?.name || labelOf(id)?.value} 权重`} value={candidate.raw[id] ?? ""} aria-invalid={Boolean(cell) || undefined} title={cell?.message} onChange={(event) => onChange(editCandidateValue(draft, candidate.id, id, event.target.value))} /></td>;
            })}
            <td className="mix-num">{entry.displacement === null ? "—" : formatUnits(entry.displacement)}</td>
            <td className="mix-status">{bad ? <span role="alert">{entry.problems.map((problem) => problem.message).join("；")}</span> : <span>有效</span>}</td>
            <td className="mix-row-actions"><div><button type="button" aria-pressed={Boolean(candidate.pinned)} aria-label={`${candidate.pinned ? "取消固定" : "固定"} ${candidate.id}`} title={candidate.pinned ? "取消固定" : "固定：重抽时保留"} onClick={() => onChange(togglePin(draft, candidate.id))}>{candidate.pinned ? <PinOff size={13} /> : <Pin size={13} />}</button>
              <button type="button" aria-label={`删除 ${candidate.id}`} title="删除此候选" onClick={() => onChange(removeCandidate(draft, candidate.id))}><Trash2 size={13} /></button></div></td>
          </tr>;
        })}</tbody>
      </table></div>
      {pages > 1 && <nav className="mix-pager" aria-label="候选分页"><button type="button" disabled={current === 0} onClick={() => setPage(current - 1)}>上一页</button><span>{current + 1} / {pages}</span><button type="button" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>下一页</button></nav>}
    </>}
    {analysis.valid && report.analysis.free.length > 0 && !rows.length && <p className="mix-empty" role="status">尚无候选。点击「生成候选预览」按偏好与覆盖率抽取（算法固定、种子可复现）。</p>}
  </section>;
}
