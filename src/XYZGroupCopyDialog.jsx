import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { Loader2, X } from "lucide-react";
import { useDialogLifecycle } from "./gallery-core.js";
import { XYZ_MOUNT_LIMIT, analysisSignature } from "./xyz-group-resolution.js";

const KIND_TEXT = { bound: "新绑定当前实例", relocate: "路径已迁移", revision: "修订已更新", current: "身份有效" };
const CONFLICT_TEXT = { "same-instance": "同一实例已独立挂载", "equal-content": "内容相同的另一个副本已独立挂载" };
const precisionOf = (value) => value ?? "—";

/** Library group -> private XYZ snapshot: verify, preview, and an explicit choice. Errors stay in this dialog. */
export default function XYZGroupCopyDialog({ state, analysis, onClose, onConfirm, onRetry }) {
  const { phase, source, projection, error } = state;
  const busy = phase === "resolving" || phase === "committing";
  // Closing is always safe: nothing is written until the final synchronous edit.
  const ref = useDialogLifecycle(true, onClose, ".xyz-owned-lora-manager");
  // A disabled button drops focus; give it back to the primary action when a busy phase ends in place.
  const wasBusy = useRef(false);
  useEffect(() => {
    if (busy) { wasBusy.current = true; return; }
    if (!wasBusy.current) return;
    wasBusy.current = false;
    if (!ref.current?.contains(document.activeElement)) (ref.current?.querySelector("footer .primary:not(:disabled), footer button:not(:disabled)") || ref.current)?.focus();
  }, [busy, ref]);
  const signature = analysis ? analysisSignature(analysis) : "";
  const legacy = projection?.bindings.filter((item) => item.kind === "bound" && !item.hadHash).length || 0;
  const moved = projection?.bindings.filter((item) => item.kind === "relocate").length || 0;
  const blocked = analysis ? [...analysis.blockers, ...analysis.enableBlockers] : [];
  const removing = analysis ? new Set(analysis.conflicts.map((item) => item.existing.instanceId)).size : 0;
  const after = analysis ? (analysis.conflicts.length ? analysis.total.afterReplace : analysis.total.afterEnable) : 0;
  return createPortal(<div className="gallery-lora-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="gallery-small-dialog xyz-group-copy-dialog" ref={ref} tabIndex={-1} role="alertdialog" aria-modal="true" aria-busy={busy} aria-label="复制 LoRA 组合快照">
      <header><div><h2>复制「{source.name}」到 XYZ？</h2></div><button type="button" aria-label="关闭复制确认" onClick={onClose}><X size={18} /></button></header>
      <div className="gallery-dialog-body">
        <p>复制 {source.members.length} 项为独立快照。之后的开关、权重与组提示词只写 XYZ，真实组合库与资产库保持原样。{source.presetPrompt?.trim() ? "组提示词按来源组合仅一次；旧实际文本不会猜测删除。" : ""}</p>
        {phase === "resolving" && <p className="xyz-copy-status" role="status"><Loader2 className="spin" size={14} aria-hidden="true" />正在向资产库核验 {source.members.length} 个成员的身份与文件……</p>}
        {phase === "blocked" && <>
          <p className="xyz-copy-lead">无法复制：以下成员没有通过资产库核验。XYZ 草稿、真实组合库与资产库均未改动。</p>
          <ul className="xyz-copy-issues">{projection.issues.map((item, index) => <li key={`${item.index}-${index}`}>{item.name && <strong>{item.name}</strong>}{item.value && <code>{item.value}</code>}<span>{item.message}</span></li>)}</ul>
        </>}
        {projection?.ok && analysis && <>
          <ol className="xyz-copy-bindings" aria-label="成员绑定预览">{projection.bindings.map((item) => <li key={item.index}>
            <strong>{item.name}</strong><code>{item.value}</code>
            <span>{KIND_TEXT[item.kind]} · 权重 {String(item.weight)} · 精度 {precisionOf(item.precision)} · {item.enabled ? "启用" : "停用"}</span>
            {item.kind === "relocate" && <small>原路径 {item.from}</small>}
          </li>)}</ol>
          {legacy > 0 && <p className="xyz-copy-note">{legacy} 个成员原先没有内容哈希：确认后绑定的是资产库当前登记的实例，并不证明它与当时使用的文件逐字节相同。</p>}
          {moved > 0 && <p className="xyz-copy-note">{moved} 个成员的路径已迁移，快照保存迁移后的当前路径并保留原路径。</p>}
          <p className="xyz-copy-note">挂载总数：现有 {analysis.total.current} 个{analysis.actions.enable || analysis.actions.replace ? ` → 启用后 ${after} / ${XYZ_MOUNT_LIMIT}` : "；本次不会写入挂载"}。</p>
          {analysis.conflicts.length > 0 && <>
            <p className="xyz-copy-lead">与现有独立挂载冲突，需要你明确选择：</p>
            <ul className="xyz-copy-issues">{analysis.conflicts.map((item) => <li key={`${item.memberIndex}-${item.existing.instanceId}`}>
              <strong>{CONFLICT_TEXT[item.kind]}</strong><code>{item.existing.name || item.existing.value}</code>
              <span>现有权重 {String(item.existing.weight)}；组内成员「{source.members[item.memberIndex]?.name || source.members[item.memberIndex]?.value}」权重 {String(source.members[item.memberIndex]?.weight)}。</span>
            </li>)}</ul>
            <p className="xyz-copy-note">「替换并启用」会移除上述 {removing} 项独立挂载（含其权重与顺序），改由组内成员取代；权重不会相加。「仅复制」保留现状，快照停用且不写入挂载。</p>
          </>}
          {blocked.length > 0 && <ul className="xyz-copy-issues">{blocked.map((item, index) => <li key={`${item.code}-${index}`}><span>{item.message}</span></li>)}</ul>}
        </>}
        {error && <p className="xyz-copy-error" role="alert">{error}</p>}
      </div>
      <footer>
        <button type="button" onClick={onClose}>取消</button>
        {phase === "blocked" && <button type="button" onClick={onRetry}>重新核验</button>}
        {phase !== "blocked" && projection?.ok && analysis && <>
          <button type="button" disabled={busy || !analysis.actions["copy-only"]} onClick={() => onConfirm("copy-only", signature)}>仅复制（不启用）</button>
          {analysis.actions.replace
            ? <button type="button" className="primary" disabled={busy} onClick={() => onConfirm("replace", signature)}>替换 {removing} 项并启用</button>
            : <button type="button" className="primary" disabled={busy || !analysis.actions.enable} onClick={() => onConfirm("enable", signature)}>{phase === "committing" ? "正在确认……" : "复制并启用快照"}</button>}
        </>}
      </footer>
    </section>
  </div>, document.body);
}
