import { ArrowDown, ArrowUp, ChevronDown, ChevronUp, Layers, ListChecks, Plus, SlidersHorizontal, Trash2 } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import XYZAxisLoraPicker from "./XYZAxisLoraPicker.jsx";
import { applySelection } from "./lora-mix-model.js";
import { PREFERENCES, PREFERENCE_LABELS, formatUnits } from "./lora-mix-rules.js";

const refForPicker = (member) => ({ ...member.ref, weight: Number(member.rule.base) || 1, precision: 4, enabled: true });
const nameOf = (member) => member.ref.name || member.ref.value;

/** The compact explicit sequence: a few leading values, an ellipsis and the end. Never the whole 100k-value list. */
function sequenceText(sequence) {
  const at = (index) => formatUnits(sequence.at(index));
  const body = sequence.size <= 8 ? Array.from({ length: sequence.size }, (_, i) => at(i)).join("、") : `${[0, 1, 2].map(at).join("、")} … ${[sequence.size - 2, sequence.size - 1].map(at).join("、")}`;
  return `${sequence.size} 项：${body}${sequence.baseOnGrid ? "" : `（含基础值 ${formatUnits(sequence.base)} 作为额外项）`}`;
}

/** One line per member that ties X1 to X2: its rule at a glance, or what still blocks it. */
function ruleSummary(member, row, roleIssue) {
  const rule = member.rule;
  if (roleIssue) return { tone: "error", text: "与全局 LoRA 重叠", title: roleIssue.message };
  if (row?.errors?.base) return { tone: "warn", text: "X2 基础权重待填写", title: row.errors.base };
  if (rule.locked === true) return { tone: "", text: `锁定 · 基础 ${rule.base}`, title: "只使用基础权重" };
  if (row && Object.values(row.errors || {}).some(Boolean)) return { tone: "warn", text: "X2 规则待修正", title: Object.values(row.errors).filter(Boolean)[0] };
  return { tone: "", text: `基础 ${rule.base} · ${rule.min} … ${rule.max} / ${rule.step}${row?.sequence ? ` · ${row.sequence.size} 项` : ""}`, title: "在 X2 中编辑规则" };
}

/**
 * X1: a selection transaction over the mother manager. It picks identities only; weights belong to X2. The card takes
 * the canvas card's height; its member list is the only part that scrolls.
 */
export function LoraMixSelection({ draft, manager, analysis, issues = [], onEditRule, onChange }) {
  const members = draft.mix.members;
  const setMembers = (next) => onChange({ ...draft, mix: { ...draft.mix, members: next } });
  const move = (index, delta) => {
    const to = index + delta;
    if (to < 0 || to >= members.length) return;
    const next = members.slice();
    [next[index], next[to]] = [next[to], next[index]];
    setMembers(next);
  };
  const editRule = (index) => {
    if (onEditRule) { onEditRule(index); return; }
    const field = document.getElementById(`mix-rule-${index}-base`);
    field?.scrollIntoView({ block: "center", behavior: "smooth" });
    field?.focus({ preventScroll: true });
  };
  return <section className="mix-x1 mix-card" aria-label="X1 选择测试 LoRA">
    <div className="section-heading"><span>X1</span><h2>选择测试 LoRA</h2><small className="mix-heading-meta">{members.length ? `已选 ${members.length}` : "未选择"}</small></div>
    <p className="mix-hint">只选成员：权重与范围在 X2 设置。左侧「全局 LoRA」是整轮固定背景，不参与测试，也不能与测试成员重叠。</p>
    <XYZAxisLoraPicker draft={draft} manager={manager} members={members.map(refForPicker)} multiple label="X1 测试 LoRA 选择"
      text={members.length ? <><ListChecks size={14} aria-hidden="true" />管理测试 LoRA…</> : <><Plus size={14} aria-hidden="true" />选择测试 LoRA…</>}
      onSelect={(picked) => onChange({ ...draft, mix: applySelection(draft.mix, picked) })} />
    <div className={`mix-member-scroll${members.length ? "" : " empty"}`} role="region" aria-label="已选测试 LoRA 列表" tabIndex={members.length ? 0 : undefined}>
      {members.length ? <ol className="mix-member-list" aria-label="已选测试 LoRA（按顺序追加到全局背景之后）">{members.map((member, index) => {
        const summary = ruleSummary(member, analysis?.rows?.[index], issues.find((issue) => issue.member === index && ["role", "members"].includes(issue.scope)));
        return <li key={member.ref.instanceId || member.ref.value} className={summary.tone ? `has-${summary.tone}` : ""}>
          <span className="mix-member-index">{index + 1}</span>
          <span className="mix-member-name"><strong title={nameOf(member)}>{nameOf(member)}</strong><code title={member.ref.value}>{member.ref.value}</code>
            <button type="button" className={`mix-member-rule${summary.tone ? ` ${summary.tone}` : ""}`} title={summary.title} aria-label={`${nameOf(member)}：${summary.text}，在 X2 中编辑`} onClick={() => editRule(index)}>{summary.text}</button></span>
          <span className="mix-member-actions" role="group" aria-label={`${nameOf(member)} 顺序与移除`}>
            <button type="button" aria-label={`上移 ${nameOf(member)}`} title="上移" disabled={index === 0} onClick={() => move(index, -1)}><ArrowUp size={13} /></button>
            <button type="button" aria-label={`下移 ${nameOf(member)}`} title="下移" disabled={index === members.length - 1} onClick={() => move(index, 1)}><ArrowDown size={13} /></button>
            <button type="button" aria-label={`移除 ${nameOf(member)}`} title="移除" onClick={() => setMembers(members.filter((_, i) => i !== index))}><Trash2 size={13} /></button>
          </span>
        </li>;
      })}</ol>
        : <div className="mix-member-empty" role="status"><Layers size={22} aria-hidden="true" /><strong>测试 LoRA 列表为空</strong>
          <span>点击上方「选择测试 LoRA…」，在 LoRA 管理器中勾选要混合的成员；选中的 LoRA 会按顺序列在这里。</span>
          <small>启用的 LoRA 合计最多 16 个（含左侧全局背景）</small></div>}
    </div>
  </section>;
}

function RuleField({ id, label, error, labelId, children }) {
  return <div className={`mix-field${error ? " has-error" : ""}`}>{labelId ? <span className="mix-label" id={labelId}>{label}</span> : <label htmlFor={id}>{label}</label>}{children}{error && <small role="alert" id={`${id}-error`}>{error}</small>}</div>;
}

/** X2: raw text is stored verbatim, errors sit beside their field, and nothing is defaulted on blur. */
/** X2 is always on the page (an empty frame until X1 has members) and can be collapsed to its heading. */
export function LoraMixRules({ draft, analysis, issues, open = true, onToggle, onChange }) {
  const members = draft.mix.members;
  const edit = (index, patch) => onChange({ ...draft, mix: { ...draft.mix, members: members.map((member, i) => i === index ? { ...member, rule: { ...member.rule, ...patch } } : member) } });
  const pending = members.filter((_, index) => Object.values(analysis.rows[index]?.errors || {}).some(Boolean) || issues.some((issue) => issue.member === index && ["role", "members"].includes(issue.scope))).length;
  return <section className={`mix-x2 mix-card${open ? "" : " collapsed"}`} aria-label="X2 成员规则">
    <div className="section-heading"><span>X2</span><h2>成员规则</h2>
      <small className={`mix-heading-meta${pending ? " warn" : ""}`}>{members.length ? `${members.length} 个成员${pending ? ` · ${pending} 个待修正` : " · 已就绪"}` : "等待 X1 选择"}</small>
      {onToggle && <button type="button" className="mix-collapse" aria-expanded={open} aria-controls="mix-x2-body" title={open ? "收起 X2（规则保留）" : "展开 X2"} onClick={() => onToggle(!open)}>{open ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}{open ? "收起" : "展开"}</button>}</div>
    {open && <div className="mix-x2-body" id="mix-x2-body">
    <p className="mix-hint">范围 −5 … 5，最多 4 位小数，步长必须大于 0；非法输入会保留并标红，不会取整或截断。锁定的成员只使用基础权重，仍属于测试集合并计入审计记录。</p>
    {!members.length && <div className="mix-rule-empty" role="status">
      <div className="mix-rule-ghost" aria-hidden="true">{["基础权重", "最小值", "最大值", "步长", "偏好"].map((label) => <span key={label}><i>{label}</i><b /></span>)}</div>
      <p><SlidersHorizontal size={14} aria-hidden="true" />在 X1 选择测试 LoRA 后，每个成员会在这里出现一行规则：基础权重、范围（最小 / 最大）、步长与偏好。</p>
    </div>}
    {members.length > 0 && <ol className="mix-rule-list">{members.map((member, index) => {
      const row = analysis.rows[index], errors = row.errors, idPrefix = `mix-rule-${index}`, locked = member.rule.locked === true, name = nameOf(member);
      const roleIssues = issues.filter((issue) => issue.member === index && ["role", "members"].includes(issue.scope));
      const text = (key, label, disabled = false) => {
        const error = !(disabled && key !== "base") && errors[key];
        return <RuleField id={`${idPrefix}-${key}`} label={label} error={error}><input id={`${idPrefix}-${key}`} type="text" inputMode="decimal" value={member.rule[key]} disabled={disabled} aria-invalid={Boolean(error) || undefined} aria-describedby={error ? `${idPrefix}-${key}-error` : undefined} onChange={(event) => edit(index, { [key]: event.target.value })} /></RuleField>;
      };
      const preferenceError = !locked && errors.preference;
      return <li key={member.ref.instanceId || member.ref.value} className={`mix-rule${locked ? " locked" : ""}`}>
        <header>
          <span className="mix-member-index">{index + 1}</span>
          <strong title={member.ref.value}>{name}</strong>
          {locked ? <span className="mix-badge">已锁定 · 固定基础</span> : row.sequence && <span className="mix-badge">{sequenceText(row.sequence)}</span>}
          <label className="mix-lock" htmlFor={`${idPrefix}-lock`}><input id={`${idPrefix}-lock`} type="checkbox" checked={locked} aria-label={`锁定 ${name}`} onChange={(event) => edit(index, { locked: event.target.checked })} />锁定</label>
        </header>
        <div className="mix-rule-fields">
          {text("base", "基础权重")}
          {text("min", "最小值", locked)}
          {text("max", "最大值", locked)}
          {text("step", "步长", locked)}
          <RuleField id={`${idPrefix}-preference`} labelId={`${idPrefix}-preference-label`} label="偏好" error={preferenceError}>
            <WorkspaceSelect ariaLabel={`${name} 偏好`} value={member.rule.preference} disabled={locked} ariaInvalid={Boolean(preferenceError) || undefined} ariaDescribedBy={preferenceError ? `${idPrefix}-preference-error` : undefined}
              options={[...(PREFERENCES.includes(member.rule.preference) ? [] : [{ value: member.rule.preference, label: `${String(member.rule.preference)}（无效）` }]), ...PREFERENCES.map((value) => ({ value, label: PREFERENCE_LABELS[value] }))]}
              onChange={(preference) => edit(index, { preference })} />
          </RuleField>
        </div>
        {roleIssues.map((issue, i) => <p className="mix-role-error" role="alert" key={i}>{issue.message}</p>)}
      </li>;
    })}</ol>}
    </div>}
  </section>;
}
