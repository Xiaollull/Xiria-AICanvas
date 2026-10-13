import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Wrench, X, Search, RefreshCw, Check, Undo2, ArrowRight, ChevronDown, RotateCcw, ListTree, Trash2, History } from "lucide-react";
import { createLoraRepairController, initialRepairState } from "./lora-repair-controller.js";
import { LORA_PUBLICATION_EVENT, startLoraPublicationWatch } from "./lora-library-client.js";
import { READY_LORA_ENGINES } from "./lora-model-scope.js";
import { useDialogLifecycle } from "./gallery-core.js";
import { relinkDraftSummary } from "./lora-relink.js";
import { describeRepairFile, repairBrowseTree, repairFolderChoices, repairKey, repairPickerSections, repairRootLabels, sameRepairTarget } from "./lora-repair-picker.js";

const undoReasons = {
  "association-edited-after-repair": "该关联或确认标签已被后来编辑，保持当前值",
  "owner-moved-or-edited": "使用者已重排、删除或改变，无法安全定位原关联",
  "card-association-edited": "卡片实例关联已改变，保持当前卡片与资料",
  "card-engine-or-owner-changed": "画廊卡片已删除或切换引擎，未恢复旧卡片",
  "tag-confirmation-edited": "精确标签确认已被后来修改，保持当前确认",
  "history-rule-edited": "历史替代规则已被后来修改，保持当前规则",
};
// Server reason codes -> short, explainable text. No confidence percentages.
const reasonText = {
  "filesystem-identity": "文件标识唯一吻合", "same-content": "完整 SHA-256 相同", "different-content": "完整 SHA-256 不同",
  "same-name": "文件名完全相同", "existing-copy": "已登记的另一份副本", "same-root": "同一根目录", "other-root": "其他根目录",
  "same-folder": "相同文件夹", "descendant-folder": "原文件夹的子文件夹", "ancestor-folder": "原文件夹的上级", "same-category": "同一类别目录",
  "same-folder-name": "同名文件夹", "other-folder": "其他文件夹", "provenance-unconfirmed": "来源未确认",
};
const conflictText = { "duplicate-target": "同一列表将出现重复目标；不会去重或叠加权重", "target-user-card-exists": "目标已有自己的卡片资料" };
const evidenceText = { relocation: "位置改变", "unverified-source": "显式改绑 · 来源未验证", "equal-content": "指定替代 · 内容相同", "different-content": "指定替代 · 内容不同", "confirmed-deleted": "确认删除 · 清理关联" };

function evidenceLabel(entry) {
  if (!entry) return "";
  const reasons = entry.reasons || [];
  if (entry.basis === "proven-relocation") return "已证实位置变化";
  if (entry.basis === "equal-content") return reasons.includes("existing-copy") ? "同内容副本 · 替代" : "完整内容相同";
  if (entry.basis === "different-content") return "内容不同 · 版本替换";
  return `${reasons.includes("same-root") ? "同根" : "其他根"}同名${reasons.includes("directory-group") ? " · 目录整体变化" : ""}`;
}
function reasonList(entry) {
  return (entry?.reasons || []).map((code) => code === "directory-group" && entry.group
    ? `${entry.group.count}/${entry.group.of} 个来源同在 ${entry.group.to || "根目录"}/` : reasonText[code]).filter(Boolean);
}
function scopeSummary(row) {
  const { scope } = row;
  if (scope.all) return `全部使用位置${scope.history ? " · 历史回填" : ""}${scope.confirmTags ? " · 改写标签" : ""}`;
  const parts = [scope.mounts && "挂载", scope.groupIds?.length && `${scope.groupIds.length} 组`, scope.galleryIds?.length && `${scope.galleryIds.length} 画廊`].filter(Boolean);
  return parts.length ? `部分：${parts.join(" · ")}` : "未选择使用位置";
}
const impactCount = (impacts) => (impacts?.mounts || 0) + (impacts?.groups?.length || 0) + (impacts?.cards?.length || 0);

export function LoraRepairEntry({ engine, locked = false }) {
  const [open, setOpen] = useState(false), button = useRef();
  return <><section className="lora-repair-entry" aria-label="LoRA 位置修复操作">
    <div><strong>LoRA 位置与关联</strong><span>扫描 → 编辑匹配 → 确认生效；不会搬动权重</span></div>
    <button ref={button} type="button" className="lora-repair-primary" onClick={() => setOpen(true)}><Wrench size={16} aria-hidden="true" />修复 LoRA 位置</button>
  </section>{open && <LoraRepairPanel engine={engine} locked={locked} onClose={() => { setOpen(false); button.current?.focus(); }} />}</>;
}

/** Anchored, portalled target picker: relevant first, every legal file reachable. */
function RepairTargetPicker({ row, observations, domain, labels, anchor, onSelect, onSkip, onClose }) {
  const [query, setQuery] = useState(""), [mode, setMode] = useState("relevant"), [place, setPlace] = useState("");
  const [active, setActive] = useState(0), [position, setPosition] = useState(null);
  const panel = useRef(), input = useRef(), id = useId();
  const sections = useMemo(() => repairPickerSections(row, observations, { query }), [row, observations, query]);
  const tree = useMemo(() => mode === "browse" ? repairBrowseTree(observations, row.engine, labels) : [], [mode, observations, row.engine, labels]);
  const groups = useMemo(() => {
    if (mode === "browse") {
      const text = query.trim().toLowerCase();
      const browsed = tree.filter((root) => !place || place.split("\0")[0] === root.rootId).flatMap((root) => root.folders
        .filter((folder) => !place || place.split("\0")[1] === undefined || place.split("\0")[1] === folder.folder)
        .map((folder) => ({ label: `${root.label} · ${folder.folder || "根目录"}`, items: text ? folder.items.filter((item) => item.value.toLowerCase().includes(text)) : folder.items })))
        .filter((group) => group.items.length);
      // A filter never hides the current choice.
      const current = row.target && observations.find((item) => item.engine === row.engine && sameRepairTarget(item, row.target));
      return current && !browsed.some((group) => group.items.includes(current)) ? [{ label: "当前选择", items: [current] }, ...browsed] : browsed;
    }
    const list = [];
    if (sections.selected) list.push({ label: "当前选择", items: [sections.selected] });
    if (sections.recommended) list.push({ label: "推荐", items: [sections.recommended] });
    if (sections.alternatives.length) list.push({ label: "其他相关候选", items: sections.alternatives });
    if (query.trim()) list.push({ label: sections.total > sections.results.length ? `搜索结果 · 前 ${sections.results.length} / ${sections.total}` : `搜索结果 · ${sections.total}`, items: sections.results });
    return list;
  }, [mode, tree, place, query, sections, row.target, observations, row.engine]);
  const options = groups.flatMap((group) => group.items);
  useEffect(() => { setActive(0); }, [query, mode, place]);
  // The first render is empty until the anchor is measured; focus once it exists.
  useEffect(() => { input.current?.focus(); }, [mode, Boolean(position)]);
  useLayoutEffect(() => {
    const update = () => {
      if (!anchor?.isConnected) return onClose(false);
      const rect = anchor.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight;
      const width = Math.min(vw - 16, 440), left = Math.min(Math.max(8, rect.right - width), vw - width - 8);
      const below = vh - rect.bottom - 14, above = rect.top - 14;
      const down = below >= 300 || below >= above;
      setPosition({ left, width, ...(down ? { top: rect.bottom + 6 } : { bottom: vh - rect.top + 6 }), maxHeight: Math.max(220, Math.min(560, down ? below : above)) });
    };
    update();
    window.addEventListener("resize", update); window.addEventListener("scroll", update, true);
    return () => { window.removeEventListener("resize", update); window.removeEventListener("scroll", update, true); };
  }, [anchor]);
  useEffect(() => {
    const outside = (event) => { if (!panel.current?.contains(event.target) && !anchor?.contains(event.target)) onClose(false); };
    document.addEventListener("mousedown", outside, true);
    return () => document.removeEventListener("mousedown", outside, true);
  }, [anchor]);
  useEffect(() => { panel.current?.querySelector(`#${CSS.escape(`${id}-option-${active}`)}`)?.scrollIntoView({ block: "nearest" }); }, [active, id]);
  const keydown = (event) => {
    // Nested layer: Escape and Tab never reach the repair dialog or the LoRA manager.
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); onClose(true); return; }
    if (event.key === "Tab") {
      event.stopPropagation();
      const controls = [...panel.current.querySelectorAll("button:not(:disabled), input:not(:disabled), select:not(:disabled)")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
      return;
    }
    if (event.target !== input.current) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); event.stopPropagation(); setActive((value) => options.length ? (value + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length : 0); }
    else if (event.key === "Home" || event.key === "End") { event.preventDefault(); setActive(event.key === "Home" ? 0 : Math.max(0, options.length - 1)); }
    else if (event.key === "Enter") { event.preventDefault(); event.stopPropagation(); if (options[active]) onSelect(options[active]); }
  };
  if (!position) return null;
  let index = -1;
  const folders = mode === "browse" ? repairFolderChoices(observations, [row.engine], labels) : [];
  return createPortal(<div className="lora-repair-picker" ref={panel} role="dialog" aria-label={`选择 ${describeRepairFile({ value: row.sourceValue, rootId: row.rootId }, labels).name} 的目标`}
    style={{ left: position.left, width: position.width, top: position.top, bottom: position.bottom, maxHeight: position.maxHeight }} onKeyDown={keydown}>
    <div className="lora-repair-picker-search"><Search size={15} aria-hidden="true" />
      <input ref={input} className="lora-repair-search-input" role="combobox" aria-expanded="true" aria-controls={`${id}-list`} aria-autocomplete="list" aria-label="搜索目标文件名或完整引用"
        aria-activedescendant={options[active] ? `${id}-option-${active}` : undefined} value={query} onChange={(event) => setQuery(event.target.value)}
        placeholder={mode === "browse" ? "在全部文件中筛选" : "搜索其他文件…"} /></div>
    {domain && <p className="lora-repair-picker-domain">只列出目标目录内的文件：{domain}（含子文件夹）</p>}
    {mode === "browse" && <label className="lora-repair-picker-place"><span>位置</span><select value={place} onChange={(event) => setPlace(event.target.value)}>
      <option value="">全部根目录与文件夹</option>
      {[...new Map(folders.map((item) => [item.rootId, item.rootLabel])).entries()].map(([rootId, label]) => <option key={rootId} value={rootId}>{label} · 全部</option>)}
      {folders.map((item) => <option key={`${item.rootId}/${item.folder}`} value={`${item.rootId}\0${item.folder ? `${item.folder}/` : ""}`}>{item.rootLabel} · {item.folder || "根目录"} ({item.files})</option>)}
    </select></label>}
    <div className="lora-repair-picker-list" id={`${id}-list`} role="listbox" aria-label="候选目标">
      {groups.map((group) => <div role="group" aria-label={group.label} key={group.label}><div className="lora-repair-picker-heading" aria-hidden="true">{group.label}</div>
        {group.items.map((item) => {
          index += 1;
          const file = describeRepairFile(item, labels), ranked = sections.ranked.get(repairKey(item)), current = sameRepairTarget(item, row.target), at = index;
          return <div key={`${group.label}/${file.key}`} id={`${id}-option-${at}`} role="option" aria-selected={current} className={`lora-repair-option${at === active ? " is-active" : ""}`}
            onMouseDown={(event) => event.preventDefault()} onMouseMove={() => setActive(at)} onClick={() => onSelect(item)}>
            <span className="lora-repair-option-main"><strong title={file.name}>{file.displayName}</strong>{current && <Check size={14} aria-label="当前选择" />}<em>{file.rootLabel}</em></span>
            <span className="lora-repair-option-sub">{file.folder || "根目录"}{ranked ? ` · ${evidenceLabel(ranked)}` : ""}{ranked?.reasons?.includes("provenance-unconfirmed") ? " · 来源未确认" : ""}</span>
          </div>;
        })}</div>)}
      {!options.length && <p className="lora-repair-picker-empty">{query.trim() ? "没有符合的文件。可改为浏览全部。" : mode === "browse" ? "该引擎没有可用文件。" : "没有相关候选。可搜索或浏览全部文件。"}</p>}
    </div>
    <div className="lora-repair-picker-actions">
      <button type="button" onClick={() => { setMode(mode === "browse" ? "relevant" : "browse"); setPlace(""); }}><ListTree size={15} aria-hidden="true" />{mode === "browse" ? "返回相关候选" : domain ? "浏览目标目录" : "浏览全部"}</button>
      <button type="button" onClick={onSkip}>跳过此行</button>
      <button type="button" onClick={() => onClose(true)}>取消</button>
    </div>
  </div>, document.body);
}

function RepairRow({ row, scan, pool, domainText, labels, busy, onEdit, onRestore, onPick, onDelete, conflicts }) {
  const [open, setOpen] = useState(false);
  const detailsId = useId();
  const source = describeRepairFile({ value: row.sourceValue, rootId: row.rootId }, labels);
  const deleting = row.target?.kind === "delete";
  const chosen = row.target && !deleting && scan.observations.find((item) => item.engine === row.engine && sameRepairTarget(item, row.target));
  const target = chosen && describeRepairFile(chosen, labels);
  const ranked = row.target && (row.candidates || []).find((candidate) => sameRepairTarget(candidate, row.target));
  const scope = row.scope;
  const toggle = (key, id, enabled) => onEdit({ scope: { ...scope, all: false, [key]: enabled ? [...new Set([...(scope[key] || []), id])] : (scope[key] || []).filter((item) => item !== id) } });
  const tags = ["SD", "iL"].includes(row.engine);
  const status = row.status === "unobserved" ? "来源根目录未扫描" : row.status === "identity-conflict" ? "原位置已被其他文件占用" : row.status === "incomplete" ? "扫描不完整" : "";
  const pending = row.suggestion?.state === "tied" ? `${row.suggestion.ties} 个同等候选 · 请选择` : row.suggestion?.state === "contested" ? "与其他来源争用同一目标 · 请选择"
    : row.suggestion?.state === "blocked" ? "需复核后手动选择" : row.candidateCount ? "仅有内容不同的同名文件" : "未找到同名或同内容文件";
  const label = deleting ? "确认已删除 · 清理关联" : row.target ? row.chosenBy === "user" ? "手动选择" : `推荐 · ${evidenceLabel(row.recommendation)}` : row.skipped ? "已跳过" : "待选择";
  const deletable = scan.complete && ["missing", "identity-conflict"].includes(row.status);
  const relocatable = Boolean(chosen && row.sourceContentHash && row.sourceContentHash === chosen.contentHash && !ranked?.existing);
  return <article className={`lora-repair-row${row.skipped ? " is-skipped" : ""}${!row.target && !row.skipped ? " is-open" : ""}`} aria-label={`来源 ${source.name}`}>
    <div className="lora-repair-cells">
      <label className="lora-repair-check"><input type="checkbox" checked={!row.skipped} disabled={busy} onChange={(event) => onEdit({ skipped: !event.target.checked })} aria-label={`修复 ${source.name}`} /></label>
      <div className="lora-repair-cell lora-repair-from"><strong title={source.relative}>{source.displayName}</strong><span>{source.rootLabel} · {source.folder || "根目录"}{status ? ` · ${status}` : ""}</span></div>
      <ArrowRight className="lora-repair-arrow" size={16} aria-hidden="true" />
      <div className="lora-repair-cell lora-repair-to">{row.skipped ? <><strong className="is-muted">已跳过</strong><span>不改变此来源的关联</span></>
        : deleting ? <><strong className="is-danger">确认已删除</strong><span>只移除所选使用位置中的此 LoRA；不删除任何文件</span></>
        : target ? <><strong title={target.relative}>{target.displayName}</strong><span>{target.rootLabel} · {target.folder || "根目录"}</span></>
        : <><strong className="is-muted">未选择</strong><span>{pending}</span></>}</div>
      <div className="lora-repair-cell lora-repair-why"><strong><b className={`lora-repair-badge${deleting ? " is-warning" : row.target && row.chosenBy === "suggestion" ? " is-suggested" : ""}`}>{label}</b>{row.needsReview && <b className="lora-repair-badge is-warning">需复核</b>}</strong>
        <span>{row.target && row.chosenBy === "suggestion" && row.recommendation?.basis !== "proven-relocation" ? "来源未确认 · " : ""}{scopeSummary(row)} · 影响 {impactCount(row.impacts)} 处</span></div>
      <div className="lora-repair-actions">
        <button type="button" disabled={busy || row.skipped} onClick={(event) => onPick(event.currentTarget)} aria-haspopup="dialog">更改</button>
        <button type="button" aria-expanded={open} aria-controls={detailsId} onClick={() => setOpen(!open)}>详情<ChevronDown size={14} aria-hidden="true" className={open ? "is-flipped" : ""} /></button>
      </div>
    </div>
    {conflicts.length > 0 && <div className="lora-repair-row-error" role="alert">{conflicts.map((item, index) => <p key={index}>{conflictText[item.reason] || item.reason}{item.galleryId ? ` · 画廊 ${item.galleryId}` : ""}；请跳过或缩小范围。</p>)}</div>}
    {open && <div className="lora-repair-details" id={detailsId}>
      <dl>
        <div><dt>原引用</dt><dd><code>{row.sourceValue}</code><small>{source.rootLabel} · {row.engine}{row.sourceContentHash ? ` · SHA-256 ${row.sourceContentHash.slice(0, 12)}…` : " · 无可信内容基线"}</small></dd></div>
        <div><dt>目标</dt><dd>{deleting ? <small>确认已删除：只清理所选范围内的关联，保留其他成员、权重、组名、备注、Prompt 与图片</small> : chosen ? <><code>{chosen.value}</code><small>{target.rootLabel}{target.shared ? ` · 根 ID ${target.rootDetail}` : ""} · SHA-256 {chosen.contentHash.slice(0, 12)}…{row.sourceContentHash ? chosen.contentHash === row.sourceContentHash ? " · 内容相同（同内容不等于同实例）" : " · 内容不同" : ""}</small></> : <small>{row.skipped ? "已跳过" : pending}</small>}</dd></div>
        {(ranked || row.recommendation) && <div><dt>依据</dt><dd><small>{reasonList(ranked || row.recommendation).join(" · ")}</small></dd></div>}
      </dl>
      <div className="lora-repair-detail-grid">
        <label>完整引用（仅接受{domainText ? "目标目录内" : "本次扫描观察到"}的文件）<input value={row.targetInput ?? (deleting ? "" : row.target?.value) ?? ""} disabled={busy || row.skipped} placeholder="style/Name.safetensors" spellCheck="false"
          onChange={(event) => { const value = event.target.value; const found = pool.find((item) => item.engine === row.engine && item.value === value); onEdit({ targetInput: value, target: found ? { value: found.value, rootId: found.rootId, kind: "replace" } : null }); }} />
          {row.targetInput && !chosen && <small className="lora-repair-invalid">{domainText ? "不在目标目录内，或未在本次扫描中观察到；如需其他位置请更改目标目录并重新扫描。" : "未在本次扫描中观察到；请从“更改”中选择。"}</small>}</label>
        <label>关联类型<select value={row.target?.kind || "replace"} disabled={!chosen || busy || row.skipped} onChange={(event) => onEdit({ target: { ...row.target, kind: event.target.value } })}>
          <option value="replace">指定替代 · 显式改绑，保留两套实例/资料</option>
          <option value="relocate" disabled={!relocatable && row.target?.kind !== "relocate"}>原实例位置改变 · 我确认来源</option></select></label>
        <fieldset className="lora-repair-scope" disabled={busy || row.skipped}><legend>影响范围</legend>
          <label><input type="checkbox" checked={Boolean(scope.all)} onChange={(event) => onEdit({ scope: { ...scope, all: event.target.checked } })} />全部使用位置（挂载 {row.impacts.mounts} · 组 {row.impacts.groups.length} · 画廊 {row.impacts.cards.length}）</label>
          {!scope.all && <><label><input type="checkbox" checked={Boolean(scope.mounts)} onChange={(event) => onEdit({ scope: { ...scope, mounts: event.target.checked } })} />挂载及兼容镜像</label>
            {row.impacts.groups.map((group) => <label key={group.id}><input type="checkbox" checked={scope.groupIds?.includes(group.id) || false} onChange={(event) => toggle("groupIds", group.id, event.target.checked)} />{group.name || group.id} · {group.enabled === false ? "禁用组" : "启用组"}</label>)}
            {row.impacts.cards.map((card) => <label key={card.id}><input type="checkbox" checked={scope.galleryIds?.includes(card.id) || false} onChange={(event) => toggle("galleryIds", card.id, event.target.checked)} />画廊 · {card.title || card.id}</label>)}</>}
          <label><input type="checkbox" checked={Boolean(scope.history)} disabled={!scope.all} onChange={(event) => onEdit({ scope: { ...scope, history: event.target.checked } })} />历史回填替代（仅全部范围）</label>
          {tags && <label><input type="checkbox" checked={Boolean(scope.confirmTags)} onChange={(event) => onEdit({ scope: { ...scope, confirmTags: event.target.checked } })} />我确认改写所选范围的精确 LoRA 标签；裸名称不改</label>}
        </fieldset>
      </div>
      {deletable && <p className="lora-repair-delete-note">{row.candidateCount ? `仍有 ${row.candidateCount} 个同名或同内容候选；若确认删除，只清理此来源的关联，不删除这些副本。`
        : `本次完整观察范围（${domainText || "该引擎全部已登记目录"}）内未找到同名或同内容文件；范围外未观察的位置不作为删除证据。`}</p>}
      <div className="lora-repair-detail-actions">
        {deletable && !deleting && <button type="button" className="lora-repair-danger" disabled={busy} onClick={onDelete}><Trash2 size={14} aria-hidden="true" />确认已删除并清理关联</button>}
        <button type="button" disabled={busy || (!deleting && (!row.recommendation || row.chosenBy === "suggestion"))} onClick={onRestore}><RotateCcw size={14} aria-hidden="true" />{deleting ? "取消删除标记" : "恢复建议目标"}</button>
      </div>
    </div>}
  </article>;
}

const SOURCE_SHORTCUTS = ["style", "character", "concept", "other"];
const lifecycleText = { settled: "已处理", history: "历史", unused: "无使用位置", archived: "已确认删除", unobserved: "未观察", incomplete: "扫描不完整" };
const domainLabel = (entries) => (entries || []).map((entry) => `${entry.label} · ${entry.engine}${entry.prefix ? ` · ${entry.prefix}/` : " · 根目录"}`).join("；");

/** P10.2 pre-scan scope: engine, problematic source folder, target folder. */
function ScopeForm({ form, setForm, directories, check, busy, scanning, onScan, onCancel }) {
  const sourceError = check?.source && !check.source.ok ? check.source.error : "", targetError = check?.target && !check.target.ok ? check.target.error : "";
  const sourceMissing = check?.source?.ok && check.source.entries?.some((entry) => !entry.exists);
  const id = useId();
  return <form className="lora-repair-scopeform" onSubmit={(event) => { event.preventDefault(); onScan(); }}>
    <label className="lora-repair-field"><span>引擎</span>
      <select value={form.engine} disabled={busy} onChange={(event) => setForm({ ...form, engine: event.target.value, target: "", targetText: "" })}><option value="all">全部引擎</option>{READY_LORA_ENGINES.map((item) => <option key={item}>{item}</option>)}</select></label>
    <div className="lora-repair-field"><label htmlFor={`${id}-source`}>问题来源目录</label>
      <select id={`${id}-source`} value={form.source} disabled={busy} title="按旧引用所在目录筛选来源；目录可以已被移走" onChange={(event) => setForm({ ...form, source: event.target.value })}>
        <option value="">全部（本地 LoRA 根目录）</option>{SOURCE_SHORTCUTS.map((item) => <option key={item} value={item}>{item}/</option>)}<option value="custom">自定义路径…</option></select>
      {form.source === "custom" && <input value={form.sourceText} disabled={busy} aria-label="自定义问题来源目录" aria-invalid={Boolean(sourceError)} placeholder="例如 style/NKStyleV3 或 models/loras/anima/style" spellCheck="false" onChange={(event) => setForm({ ...form, sourceText: event.target.value })} />}
      {(sourceError || sourceMissing) && <small id={`${id}-source-hint`} className={sourceError ? "lora-repair-invalid" : ""} role={sourceError ? "alert" : undefined}>{sourceError || "该目录当前不存在；仍按已登记记录与使用中的引用查找来源。"}</small>}</div>
    <div className="lora-repair-field"><label htmlFor={`${id}-target`}>目标匹配目录</label>
      <select id={`${id}-target`} value={form.target} disabled={busy} title="包含所有子文件夹；留空 = 全库匹配" onChange={(event) => setForm({ ...form, target: event.target.value })}>
        <option value="">不限（全库匹配）</option>
        {(directories || []).filter((root) => root.enabled).map((root) => <optgroup key={`${root.engine}/${root.rootId}`} label={`${root.label} · ${root.engine}`}>
          <option value={`${root.rootId}\0`}>{root.label} 根目录</option>
          {root.directories.map((folder) => <option key={folder} value={`${root.rootId}\0${folder}`}>{folder}/</option>)}</optgroup>)}
        <option value="custom">自定义路径…</option></select>
      {form.target === "custom" && <input value={form.targetText} disabled={busy} aria-label="自定义目标匹配目录" aria-invalid={Boolean(targetError)} placeholder="例如 style 或服务器上的绝对路径" spellCheck="false" onChange={(event) => setForm({ ...form, targetText: event.target.value })} />}
      {targetError && <small id={`${id}-target-hint`} className="lora-repair-invalid" role="alert">{targetError}</small>}</div>
    <div className="lora-repair-field lora-repair-scan-actions"><span aria-hidden="true">&nbsp;</span>{scanning ? <button type="button" onClick={onCancel}><X size={15} aria-hidden="true" />取消扫描</button>
      : <button type="submit" className="lora-repair-primary" disabled={busy || Boolean(sourceError || targetError)}><RefreshCw size={15} aria-hidden="true" />扫描并匹配</button>}</div>
  </form>;
}

const formParams = (form) => {
  const engines = form.engine === "all" ? READY_LORA_ENGINES : [form.engine];
  const source = form.source === "custom" ? form.sourceText.trim() : form.source;
  const target = form.target === "custom" ? form.targetText.trim() : form.target ? { rootId: form.target.split("\0")[0], path: form.target.split("\0")[1] } : "";
  return { engines, source, target };
};

export default function LoraRepairPanel({ engine, locked, onClose }) {
  const [state, setState] = useState(initialRepairState);
  const [form, setFormState] = useState({ engine: engine || "SD", source: "", sourceText: "", target: "", targetText: "" });
  const [filter, setFilter] = useState(""), [discard, setDiscard] = useState(false), [picker, setPicker] = useState(null);
  const dialog = useDialogLifecycle(true, () => discard ? setDiscard(false) : close(), "", !["committing", "undoing"].includes(state.phase)), controller = useRef();
  if (!controller.current) controller.current = createLoraRepairController({ changed: setState });
  const api = controller.current;
  useEffect(() => {
    startLoraPublicationWatch(); void api.availability(); dialog.current?.querySelector("button")?.focus();
    const changed = (event) => api.markRevision(event.detail.revision);
    window.addEventListener(LORA_PUBLICATION_EVENT, changed);
    return () => { window.removeEventListener(LORA_PUBLICATION_EVENT, changed); api.cancel(); };
  }, [api]);
  // Folder list + inline validation follow the form (debounced); no hashing.
  useEffect(() => {
    const params = formParams(form);
    api.setParams(params);
    const timer = setTimeout(() => void api.loadScope(params), 250);
    return () => clearTimeout(timer);
  }, [api, form]);
  const busy = ["scanning", "previewing", "committing", "undoing"].includes(state.phase);
  useEffect(() => { if (discard) dialog.current?.querySelector(".lora-repair-discard button")?.focus(); }, [discard]);
  useEffect(() => { if (busy) setPicker(null); }, [busy]);
  const labels = useMemo(() => repairRootLabels(state.scan?.roots), [state.scan]);
  const close = () => { if (state.phase === "committing" || state.phase === "undoing") return; if (relinkDraftSummary(state.draft).selected && state.phase !== "result") setDiscard(true); else onClose(); };
  const keydown = (event) => {
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (discard) { setDiscard(false); dialog.current?.querySelector("header button")?.focus(); } else close(); }
    if (event.key === "Tab") {
      event.stopPropagation();
      const root = discard ? dialog.current.querySelector(".lora-repair-discard") : dialog.current;
      const controls = [...root.querySelectorAll("button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary")].filter((element) => element.getClientRects().length);
      const first = controls[0], last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  };
  const summary = relinkDraftSummary(state.draft);
  // A published (or undone) draft is consumed; only undo, rescan or close remain.
  const done = state.phase === "result";
  const rows = state.draft?.rows || [], processed = state.draft?.processed || [];
  const text = filter.trim().toLowerCase();
  const visible = text ? rows.filter((row) => row.sourceValue.toLowerCase().includes(text) || (row.target?.value || "").toLowerCase().includes(text)) : rows;
  const targetDomain = state.scan?.domain?.target ? domainLabel(state.scan.domain.target) : "";
  const pool = useMemo(() => (state.scan?.observations || []).filter((item) => !item.outsideTarget), [state.scan]);
  const conflictsFor = (row) => (state.preview?.conflicts || []).filter((item) => item.sourceId === row.sourceId);
  const looseConflicts = (state.preview?.conflicts || []).filter((item) => !rows.some((row) => row.sourceId === item.sourceId));
  const allChecked = visible.length > 0 && visible.every((row) => !row.skipped), someChecked = visible.some((row) => !row.skipped);
  const orphans = rows.filter((row) => !row.skipped && !row.target && !row.candidateCount && state.scan?.complete && ["missing", "identity-conflict"].includes(row.status));
  const blockers = !state.available ? "当前服务尚未加载完整修复契约；可扫描，不可确认。"
    : locked ? "活动、排队或恢复中的任务尚未安全空闲。" : state.staleParams ? "扫描参数已改变；请重新扫描后再检查与确认。"
    : !state.scan?.complete ? "扫描不完整；请选择可完整观察的范围后重扫。" : summary.review ? "发布版本已改变，所选行需要复核。"
    : !state.preview ? "请先检查影响与冲突。" : state.preview.conflicts?.length ? "存在冲突，请跳过相关行或缩小影响范围。" : "";
  const evidenceCounts = state.preview?.results?.reduce((counts, item) => ({ ...counts, [item.evidence]: (counts[item.evidence] || 0) + 1 }), {});
  // Only a draft with selected rows is worth a discard prompt (same rule as the controller).
  const scanNow = (discardDraft = false) => { if (summary.selected && !discardDraft && state.phase !== "result") { setDiscard("rescan"); return; } void api.scan(formParams(form), { discard: true }); };
  const statusLine = state.phase === "scanning" ? "正在核对完整内容…未改变任何关联" : !state.scan ? "选择范围后扫描；扫描只读取文件，不改变关联"
    : `${state.scan.complete ? "" : "扫描不完整 · "}${summary.sources} 个待处理 · ${summary.suggested} 个已建议${summary.manual ? ` · ${summary.manual} 个手动` : ""}${summary.deletions ? ` · ${summary.deletions} 个确认删除` : ""} · ${summary.unselected} 个未选择${summary.skipped ? ` · ${summary.skipped} 个跳过` : ""}${processed.length ? ` · 已处理/历史 ${processed.length}` : ""}`;
  const relinks = summary.selected - summary.deletions;
  return <div className="lora-repair-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
    <section className="lora-repair-panel" ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="lora-repair-title" onKeyDown={keydown}>
      <header><div><h2 id="lora-repair-title">修复 LoRA 位置</h2><p>先选范围再扫描；建议均可修改，只有最终确认后才更新选定关联，不改写旧图片或工作流。</p></div>
        <button type="button" className="lora-repair-close" aria-label="关闭修复面板" disabled={state.phase === "committing" || state.phase === "undoing"} onClick={close}><X size={18} /></button></header>
      <ScopeForm form={form} setForm={setFormState} directories={state.directories} check={state.scopeCheck} busy={busy} scanning={state.phase === "scanning"} onScan={() => scanNow()} onCancel={() => api.cancel()} />
      <div className="lora-repair-toolbar">
        <span className="lora-repair-status" role="status">{statusLine}{targetDomain ? ` · 目标：${targetDomain}（含子文件夹）` : ""}</span>
        {orphans.length > 0 && !busy && !done && <button type="button" className="lora-repair-danger" onClick={() => api.markDeleted(orphans.map((row) => row.sourceId))}><Trash2 size={15} aria-hidden="true" />{orphans.length} 个无候选来源标记为已删除…</button>}
        {rows.length > 0 && <label className="lora-repair-filter"><Search size={14} aria-hidden="true" /><input className="lora-repair-search-input" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="筛选来源或目标" aria-label="筛选来源或目标" /></label>}
      </div>
      <div className="lora-repair-body">
        {state.draft && rows.length > 0 && <div className="lora-repair-columns">
          <label className="lora-repair-check"><input type="checkbox" checked={allChecked} ref={(element) => { if (element) element.indeterminate = someChecked && !allChecked; }}
            disabled={busy || !visible.length} onChange={(event) => api.skipRows(visible.map((row) => row.sourceId), !event.target.checked)} aria-label={text ? "选择全部筛选结果" : "选择全部来源"} /></label>
          <span aria-hidden="true">原位置</span><span aria-hidden="true" /><span aria-hidden="true">目标</span><span aria-hidden="true">依据 · 范围</span><span aria-hidden="true">操作</span></div>}
        {!state.scan && state.phase !== "scanning" && <div className="lora-repair-empty"><strong>选择引擎、问题来源目录和目标目录，然后扫描并匹配</strong><p>扫描只读取文件内容，不移动权重、不改变任何挂载、组或画廊关联；确认前都可以修改或跳过。</p></div>}
        {state.phase === "scanning" && !state.draft && <div className="lora-repair-empty"><strong>正在扫描…</strong><p>首次扫描会读取所选范围内的完整文件内容；可随时取消，不会留下关联变更。</p></div>}
        {state.draft && rows.length === 0 && <div className="lora-repair-empty"><strong>没有待处理的 LoRA 位置</strong><p>{processed.length ? "已处理或仅剩历史记录的来源列在下方，不再计为待修复。" : "所选范围内的已登记与使用中引用都在原位置。"}</p></div>}
        {state.draft && rows.length > 0 && !visible.length && <div className="lora-repair-empty"><p>没有符合筛选的来源。</p></div>}
        {visible.map((row) => <RepairRow key={row.sourceId} row={row} scan={state.scan} pool={pool} domainText={targetDomain} labels={labels} busy={busy} conflicts={conflictsFor(row)}
          onEdit={(patch) => api.edit(row.sourceId, patch)} onRestore={() => api.restore(row.sourceId)} onDelete={() => api.markDeleted([row.sourceId])} onPick={(anchor) => setPicker({ sourceId: row.sourceId, anchor })} />)}
        {processed.length > 0 && <details className="lora-repair-processed"><summary><History size={14} aria-hidden="true" />已处理与历史 · {processed.length}（不计为待修复）</summary>
          <ul>{processed.map((row) => { const file = describeRepairFile({ value: row.sourceValue, rootId: row.rootId }, labels); return <li key={row.sourceId}>
            <span><strong title={row.sourceValue}>{file.displayName}</strong><small>{file.rootLabel} · {file.folder || "根目录"}</small></span>
            <b className="lora-repair-badge">{lifecycleText[row.lifecycle] || row.lifecycle}</b>
            {row.lifecycle === "unused" && !busy && !done ? <button type="button" className="lora-repair-danger" onClick={() => api.markDeleted([row.sourceId])}>确认已删除并归档</button> : <span />}
            <small>{row.lifecycle === "settled" ? `已替代为 ${row.effectiveTarget?.value}` : row.lifecycle === "history" ? `依赖 ${row.effectiveTarget?.value}（当前缺失，作为主要问题单独列出）` : row.lifecycle === "archived" ? `确认删除于 ${row.retired?.at?.slice(0, 16).replace("T", " ") || "—"}；可用该操作的撤销恢复` : row.lifecycle === "unused" ? "缺失且没有任何使用位置" : "证据不足；不视为已删除或已处理"}</small>
          </li>; })}</ul></details>}
      </div>
      {state.scan && (state.error || state.scan.issues?.length || state.preview || state.receipt || state.scan.domain?.fullRoots !== false) && <div className="lora-repair-stage">
        {state.error && <div className="lora-repair-error" role="alert">{state.error}</div>}
        {state.scan.issues?.length > 0 && <details className="lora-repair-issues"><summary>扫描完整性说明 · {state.scan.issues.length} 项（未观察不等于已删除）</summary>{state.scan.issues.map((issue, index) => <p key={index}>{issue.engine} / {labels.get(issue.rootId)?.label || issue.rootId} / {issue.relative || "根目录"}：{issue.reason}</p>)}</details>}
        {looseConflicts.length > 0 && <div className="lora-repair-error" role="alert"><strong>不能静默合并或叠加权重</strong>{looseConflicts.map((item, index) => <p key={index}>{conflictText[item.reason] || item.reason} · {(item.address || []).join(" / ") || item.galleryId}</p>)}</div>}
        {state.preview && !state.preview.conflicts?.length && state.preview.results && <p className="lora-repair-preview" role="status"><Check size={15} aria-hidden="true" />已检查：{state.preview.noOp ? "没有关联变更" : `工作区 ${state.preview.affected.workspace} 处 · 用户卡片 ${state.preview.affected.cards} · 画廊 ${state.preview.affected.gallery}`}{Object.entries(evidenceCounts || {}).map(([key, value]) => ` · ${evidenceText[key] || key} ${value} 行`).join("")}{state.registerBaseline ? " · 同时登记完整观察基线" : ""}</p>}
        {state.receipt && <section className="lora-repair-result" role="status"><Check size={18} aria-hidden="true" /><div><strong>{state.receipt.undone ? "关联撤销已发布" : state.receipt.status === "no-op" ? "无关联变更；未创建事务" : "所选关联已完整发布"}</strong>
          {state.receipt.affected && !state.receipt.undone && <p>工作区列表/标签 {state.receipt.affected.workspace} · 用户卡片 {state.receipt.affected.cards} · 画廊卡片 {state.receipt.affected.gallery}{state.receipt.results?.length ? ` · ${state.receipt.results.map((item) => evidenceText[item.evidence] || item.kind).filter((value, index, list) => list.indexOf(value) === index).join(" / ")}` : ""}</p>}
          {state.receipt.undone && <p>仅逆转仍属于本次操作的关联；后续权重、文字和资料保持。</p>}
          {state.receipt.conflicts?.map((item, index) => <p key={index}>未撤销：{undoReasons[item.reason] || item.reason} · {(item.address || []).join(" / ") || item.cardId || item.card || item.sourceId || "关联"}</p>)}
          <code>{state.receipt.operationId || "no-op"}</code></div></section>}
        {state.scan.domain?.fullRoots !== false && <label className="lora-repair-baseline"><input type="checkbox" checked={state.registerBaseline} disabled={busy || !state.scan.complete} onChange={(event) => api.baseline(event.target.checked)} />{state.scan.initial ? "同时登记当前完整观察，供以后识别位置变化" : "同时更新完整观察基线（不会自动改绑未选来源）"}</label>}
      </div>}
      <footer><p>{done ? state.receipt?.undone ? "撤销已发布；可重新扫描或关闭。" : "已发布；可撤销本次关联、重新扫描或关闭。" : blockers || `${relinks} 行改绑${summary.deletions ? ` · ${summary.deletions} 行清理关联` : ""}；权重、顺序、启用状态与未选范围保持。`}</p><div>
        <button type="button" onClick={close} disabled={state.phase === "committing" || state.phase === "undoing"}>关闭</button>
        <button type="button" onClick={() => void api.preview()} disabled={!state.draft || busy || done || state.staleParams || (!summary.selected && !state.registerBaseline)}>{state.phase === "previewing" ? "正在检查…" : "检查影响与冲突"}</button>
        <button type="button" className="lora-repair-primary" disabled={busy || done || Boolean(blockers) || (!summary.selected && !state.registerBaseline)} onClick={() => void api.commit()}>{state.phase === "committing" ? "正在完整发布…" : `确认生效${summary.selected ? ` · ${summary.selected} 行` : ""}`}</button>
        {state.receipt?.operationId && !state.receipt.undone && <button type="button" disabled={busy || locked} onClick={() => void api.undo()}><Undo2 size={15} />撤销关联</button>}
      </div></footer>
      {discard && <div className="lora-repair-discard" role="alertdialog" aria-label="未提交修复草稿"><strong>未确认的草稿不会自动应用。</strong><p>{discard === "rescan" ? "重新扫描将替换本次未提交草稿；也可以继续编辑。" : "关闭将放弃本次未提交草稿。"}</p><div><button type="button" onClick={() => setDiscard(false)}>继续编辑</button><button type="button" onClick={() => { const action = discard; setDiscard(false); api.cancel(); if (action === "rescan") scanNow(true); else onClose(); }}>放弃草稿{discard === "rescan" ? "并重扫" : "并关闭"}</button></div></div>}
    </section>
    {picker && state.draft && (() => {
      const row = rows.find((item) => item.sourceId === picker.sourceId);
      if (!row) return null;
      const done = (refocus) => { setPicker(null); if (refocus) picker.anchor?.focus(); };
      return <RepairTargetPicker row={row} observations={pool} domain={targetDomain} labels={labels} anchor={picker.anchor} onClose={done}
        onSelect={(item) => { api.edit(row.sourceId, { target: { value: item.value, rootId: item.rootId, kind: sameRepairTarget(item, row.recommendation) && row.recommendation.kind === "relocate" ? "relocate" : "replace" }, targetInput: undefined, skipped: false }); done(true); }}
        onSkip={() => { api.edit(row.sourceId, { skipped: true }); done(true); }} />;
    })()}
  </div>;
}
