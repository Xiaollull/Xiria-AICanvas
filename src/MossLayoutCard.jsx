// The Basic page's layout-mode prompt card (QPlan2/MOSS/P2 §6.1): what the layout looks like, whether
// it compiles, how many tokens it costs, and the way into the Studio.
import { useEffect, useState } from "react";
import { Check, ClipboardCopy, LayoutTemplate, PenSquare, TriangleAlert, Type, Undo2, X } from "lucide-react";
import {
  MOSS_LAYOUT_LABELS,
  MOSS_LAYOUT_LABEL_NAMES,
  MOSS_LAYOUT_MAX_TOKENS,
  MOSS_LAYOUT_TOKEN_WARNING,
  mossLayoutStats,
} from "./moss-layout.js";
import { mossLayoutPeakTokens } from "./moss-layout-tokens.js";

const formatCount = (value) => Number(value).toLocaleString("en-US");

/** `142 / 4,096` with a bar; amber from 3,600, red past the window. */
export function MossTokenMeter({ tokens, compact = false }) {
  const peak = mossLayoutPeakTokens(tokens);
  const max = tokens?.maxTokens || MOSS_LAYOUT_MAX_TOKENS;
  const level = peak === null ? "" : peak > max ? "over" : peak > MOSS_LAYOUT_TOKEN_WARNING ? "warn" : "ok";
  const label = tokens?.status === "counting" ? "计数中…"
    : tokens?.status === "error" ? "无法计数"
      : peak === null ? "—" : `${formatCount(peak)} / ${formatCount(max)}`;
  const stageTitle = tokens?.status === "ready" && tokens.stages.length
    ? `基础画布 ${formatCount(tokens.tokens)}；${tokens.stages.map((stage) => `${stage.width}×${stage.height} ${formatCount(stage.tokens)}`).join("；")}`
    : tokens?.status === "error" ? `推理服务未能计数：${tokens.error}。提交时仍会检查。` : "MOSS 最多读取 4,096 个 token（含对话模板）";
  return <div className={`moss-token-meter ${level} ${compact ? "compact" : ""}`} title={stageTitle}>
    <span>TOKENS</span>
    <i><b style={{ width: `${peak === null ? 0 : Math.min(100, peak / max * 100)}%` }} /></i>
    <output>{label}</output>
  </div>;
}

/** The boxes drawn at the canvas's aspect ratio inside whatever box holds it (the card, a template row). */
export function MossLayoutMiniature({ layout, className = "" }) {
  const { width, height } = layout;
  const unit = Math.max(width, height) / 100;
  let number = 0;
  return <svg className={`moss-layout-mini ${className}`} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <rect className="paper" width={width} height={height} />
    {layout.items.map((item) => {
      number += 1;
      if (item.hidden) return null;
      const [x1, y1, x2, y2] = item.box;
      return <g key={item.id} className={`label-${item.label}`}>
        <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} />
        <text x={x1 + unit * 1.6} y={y1 + unit * 6} fontSize={unit * 5.5}>{number}</text>
      </g>;
    })}
  </svg>;
}

export default function MossLayoutCard({ layout, compiled, tokens, prefix, positiveText, disabled, notice, onOpen, onCopy, onToText, onTemplate, onDismissNotice, onUndoNotice }) {
  const [confirmText, setConfirmText] = useState(false);
  const [copied, setCopied] = useState("");
  useEffect(() => {
    if (!confirmText) return undefined;
    const timer = setTimeout(() => setConfirmText(false), 4000);
    return () => clearTimeout(timer);
  }, [confirmText]);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = setTimeout(() => setCopied(""), 2200);
    return () => clearTimeout(timer);
  }, [copied]);

  const stats = mossLayoutStats(layout);
  const empty = !layout.items.length && !layout.summary.caption.trim();
  const issues = compiled.issues;
  const peak = mossLayoutPeakTokens(tokens);
  const over = peak !== null && peak > (tokens.maxTokens || MOSS_LAYOUT_MAX_TOKENS);
  const firstIssue = issues[0];
  const issueItem = firstIssue?.itemId ? layout.items.findIndex((item) => item.id === firstIssue.itemId) + 1 : 0;
  const copy = async () => setCopied(await onCopy() ? "已复制" : "复制失败");
  const toText = () => {
    if (positiveText.trim() && !confirmText) { setConfirmText(true); return; }
    setConfirmText(false);
    onToText();
  };

  return <section className={`moss-layout-card ${issues.length || over ? "invalid" : ""}`} aria-label="MOSS 布局提示词">
    <header>
      <span>布局提示词<b className="special-tag-mark">MOSS LAYOUT</b></span>
      <small>{layout.width} × {layout.height} · {stats.visible} 个框{stats.total > stats.visible ? `（${stats.total - stats.visible} 个已隐藏）` : ""}</small>
    </header>
    {empty ? <div className="moss-layout-card-empty">
      <button type="button" className="moss-layout-empty-canvas" onClick={onOpen} disabled={disabled} aria-label="打开布局编辑器">
        <PenSquare size={20} /><span>{layout.width} × {layout.height}</span>
      </button>
      <div>
        <strong>还没有布局</strong>
        <p>在画布上拖拽画框，给每个框写一句描述；文字框里填写要原样渲染的文字。提交时按 MOSS 官方格式编译为坐标提示词。</p>
        <div className="moss-layout-actions">
          <button type="button" className="primary" onClick={onOpen} disabled={disabled}><PenSquare size={13} />打开布局编辑器</button>
          <button type="button" onClick={() => onTemplate("readme-poster")} disabled={disabled}><LayoutTemplate size={13} />从官方示例开始</button>
        </div>
      </div>
    </div> : <div className="moss-layout-card-body">
      <button type="button" className="moss-layout-preview" onClick={onOpen} disabled={disabled} title="编辑布局">
        <MossLayoutMiniature layout={layout} />
      </button>
      <div className="moss-layout-card-info">
        <p className={`moss-layout-caption ${layout.summary.caption.trim() ? "" : "missing"}`}>{layout.summary.caption.trim() || "尚未填写全局画面描述（必填）"}</p>
        <div className="moss-layout-counts">
          {MOSS_LAYOUT_LABELS.map((label) => <span key={label} className={`label-${label} ${stats.counts[label] ? "" : "zero"}`}><i />{MOSS_LAYOUT_LABEL_NAMES[label]} {stats.counts[label]}</span>)}
        </div>
        <MossTokenMeter tokens={tokens} />
        <p className={`moss-layout-validity ${issues.length || over ? "bad" : "good"}`} role="status">
          {issues.length
            ? <><TriangleAlert size={13} />{issues.length} 处待修正：{issueItem ? `#${issueItem} ` : ""}{firstIssue.path.replace(/^items\[\d+\]\.?/, "") || "布局"} {firstIssue.message}</>
            : over ? <><TriangleAlert size={13} />提示词 {formatCount(peak)} tokens，超过 MOSS 的 {formatCount(tokens.maxTokens)} 上限；请缩短描述或删去元素</>
              : <><Check size={13} />布局有效，提交时自动编译为官方坐标格式</>}
        </p>
        {prefix && <p className="moss-layout-prefix" title={prefix}><b>前缀</b>{prefix}</p>}
        <div className="moss-layout-actions">
          <button type="button" className="primary" onClick={onOpen} disabled={disabled}><PenSquare size={13} />编辑布局</button>
          <button type="button" onClick={copy} disabled={!compiled.prompt}><ClipboardCopy size={13} />{copied || "复制提示词"}</button>
          <button type="button" className={confirmText ? "confirm" : ""} onClick={toText} disabled={disabled || !compiled.prompt} title="把编译后的布局 JSON 写入文本提示词，并切回文本模式"><Type size={13} />{confirmText ? "覆盖文本框中的内容？" : "转为文本"}</button>
        </div>
      </div>
    </div>}
    {notice && <div className="moss-layout-notice" role="status">
      <span>{notice.message}</span>
      {notice.undo && <button type="button" onClick={onUndoNotice}><Undo2 size={12} />撤销</button>}
      <button type="button" className="icon" onClick={onDismissNotice} aria-label="关闭提示"><X size={12} /></button>
    </div>}
  </section>;
}
