import { Check, ExternalLink, FileText, Palette, Plus, RefreshCw, Search } from "lucide-react";

export default function LoraLibraryCard({ item, categoryLabel, displayName = "", mounted, locked, axisMarks = [], presentation: base, lookup, onToggle, onLookup, onCustomize, onDetails }) {
  const p = displayName && !base.renamed ? { ...base, title: displayName } : base;
  const metadata = item.metadata;
  return <article aria-disabled={locked} className={`lora-library-card ${mounted ? "mounted" : ""} ${p.customized ? "customized" : ""} ${locked ? "locked" : ""}`}
    role="button" tabIndex="0" onClick={() => !locked && onToggle()} onKeyDown={(event) => { if (event.target === event.currentTarget && ["Enter", " "].includes(event.key)) { event.preventDefault(); if (!locked) onToggle(); } }}>
    <div className="lora-card-preview"><span>No image</span>{p.coverUrl && <img src={p.coverUrl} alt="" onError={(event) => event.currentTarget.remove()} />}
      <button className="lora-card-query" disabled={lookup?.loading} title="使用文件 SHA-256 在 Civitai.com 和 Civitai.red 查询" onClick={(event) => { event.stopPropagation(); onLookup(); }}>{lookup?.loading ? <RefreshCw className="spin" size={13} /> : <Search size={13} />}{lookup?.loading ? "查询中" : "查询来源"}</button>
      <button className="lora-card-customize" disabled={!onCustomize} title={onCustomize ? "自定义封面、提示词和显示信息" : "轴选择不编辑共享卡片"} aria-label={`自定义 ${item.name} 的卡片`} onClick={(event) => { event.stopPropagation(); onCustomize?.(); }}><Palette size={13} />自定义</button>
      <span className="lora-card-state">{mounted ? <><Check size={14} />已挂载</> : <><Plus size={14} />挂载</>}</span>
      {axisMarks.length > 0 && <span className="lora-card-axis-mark"><b>{axisMarks.join('、')}</b> 轴已选</span>}
    </div>
    <div className="lora-card-copy"><strong title={p.renamed ? `${p.title}（${item.name}）` : item.name}>{p.title}</strong><small>{(item.size / 1024 / 1024).toFixed(1)} MB · {categoryLabel}</small>
      {p.tags.length > 0 && <div className="lora-card-tags">{p.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}
      {metadata?.status === "found" && <div className="lora-card-metadata"><span title={`${metadata.modelName} · ${metadata.versionName}`}>{metadata.modelName} · {metadata.versionName}</span><small>{metadata.baseModel}{metadata.triggerGroups?.length ? ` · ${metadata.triggerGroups.flatMap((group) => group.words).slice(0, 3).join(", ")}` : ""}</small><a href={metadata.sourceUrl} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>{metadata.sourceDomain}<ExternalLink size={11} /></a></div>}
      {(metadata?.status === "not_found" || lookup?.error) && <span className="lora-card-error">{lookup?.error || "未找到来源"}</span>}
      <button type="button" className="lora-card-detail" onClick={(event) => { event.stopPropagation(); onDetails(); }}><FileText size={12} />查看详细信息</button>
    </div>
  </article>;
}
