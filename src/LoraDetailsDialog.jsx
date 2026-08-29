import { useEffect, useRef, useState } from "react";
import { Copy, ExternalLink, FileText, RefreshCw, X } from "lucide-react";

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index < 2 ? 0 : 2)} ${units[index]}`;
}

async function copyText(value) {
  if (!value) return false;
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  }
}

function metadataRows(metadata) {
  const local = metadata?.localMetadata || {};
  return [
    ["Civitai 模型", metadata?.modelName],
    ["模型版本", metadata?.versionName],
    ["模型类型", metadata?.modelType],
    ["基础模型", metadata?.baseModel || local.baseModel],
    ["作者", metadata?.creator || local.author],
    ["本地标题", local.title],
    ["本地格式", local.format],
    ["模型架构", local.architecture],
    ["网络实现", local.implementation],
    ["网络参数", local.network],
    ["训练分辨率", local.resolution],
    ["训练图片", local.trainingImages],
    ["训练轮数", local.epochs],
    ["训练步数", local.steps],
    ["优化器", local.optimizer],
    ["元数据日期", local.date],
    ["文件体积", formatBytes(metadata?.fileSize)],
    ["SHA-256", metadata?.hash],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function detailText(item, metadata) {
  const rows = metadataRows(metadata).map(([label, value]) => `${label}: ${value}`);
  const triggers = metadata?.triggerGroups?.length
    ? metadata.triggerGroups.flatMap((group) => [`${group.label}:`, group.words.join(", ")])
    : ["触发词: 无"];
  const tags = metadata?.triggerReviewKind !== "style" && metadata?.localMetadata?.topTags?.length
    ? ["训练高频标签:", metadata.localMetadata.topTags.map((tag) => `${tag.word} (${tag.count})`).join(", ")]
    : [];
  return [item.name, ...rows, ...triggers, ...tags].join("\n");
}

function CopyButton({ value, label = "复制" }) {
  const [copied, setCopied] = useState(false);
  return <button type="button" className="lora-detail-copy" disabled={!value} onClick={async () => {
    const success = await copyText(value);
    setCopied(success);
    if (success) window.setTimeout(() => setCopied(false), 1500);
  }}><Copy size={13} />{copied ? "已复制" : label}</button>;
}

export default function LoraDetailsDialog({ item, metadata, loading = false, error = "", onRefresh, onClose }) {
  const dialogRef = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previousFocus = document.activeElement;
    const focusableSelector = 'button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])';
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = [...(dialogRef.current?.querySelectorAll(focusableSelector) || [])].filter((element) => element.getClientRects().length > 0);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown, true);
    const frame = window.requestAnimationFrame(() => dialogRef.current?.querySelector("button")?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("keydown", handleKeyDown, true);
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
    };
  }, []);

  const found = metadata?.status === "found";
  const local = metadata?.localMetadata || {};
  const rows = metadataRows(metadata);
  const triggerGroups = Array.isArray(metadata?.triggerGroups) ? metadata.triggerGroups : [];
  const promptReview = metadata?.promptReview;
  const reviewSources = Array.isArray(promptReview?.reviewedSources) ? promptReview.reviewedSources : [];
  const styleReview = metadata?.triggerReviewKind === "style";
  const versionReviewText = promptReview?.versionScopeKind === "legacy-section"
    ? `本地版本为 ${metadata?.versionName || "旧版"}；作者未细分各旧版本，采用其共用旧版区。`
    : promptReview?.versionScopeKind === "ambiguous"
      ? `检测到新版/旧版分区，但无法确认本地版本归属，因此未从简介提取跨版本 Prompt。`
      : promptReview?.versionScope
        ? `简介只采用与本地文件匹配的 ${promptReview.versionScope} 章节，其他版本已排除。`
        : "";
  const sourceLabel = (source) => ({ "civitai-detail": "Civitai Detail", "civitai-description": "Civitai 简介", "civitai-trained-words": "Civitai 触发词", safetensors: "本地模型元数据" }[source] || "已审查来源");
  return <div className="lora-detail-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="lora-detail-dialog" role="dialog" aria-modal="true" aria-labelledby="lora-detail-title" tabIndex="-1">
      <header className="lora-detail-head">
        <div><span>LORA ASSET DOSSIER</span><h2 id="lora-detail-title">LoRA 详细信息</h2><p title={item.value}>{item.name}</p></div>
        <div><button type="button" disabled={loading} onClick={onRefresh}><RefreshCw className={loading ? "spin" : ""} size={15} />{loading ? "解析中" : "重新解析"}</button><button type="button" aria-label="关闭 LoRA 详细信息" onClick={onClose}><X size={19} /></button></div>
      </header>
      <div className="lora-detail-status">
        <span className={found ? "found" : "local"}><i />{found ? `已连接 ${metadata.sourceDomain}` : metadata ? "本地元数据" : "等待解析"}</span>
        {metadata?.queriedAt && <small>查询时间 {new Date(metadata.queriedAt).toLocaleString("zh-CN")}</small>}
        {found && <a href={metadata.sourceUrl} target="_blank" rel="noreferrer">打开来源页<ExternalLink size={12} /></a>}
        <CopyButton value={metadata ? detailText(item, metadata) : ""} label="复制全部" />
      </div>
      <div className="lora-detail-body">
        <main>
          <section className="lora-detail-section lora-trigger-section">
            <header><div><span>ACTIVATION GUIDE</span><h3>{styleReview ? "风格触发词" : "触发词与特征组合"}</h3></div><small>{styleReview ? "风格 LoRA 只显示明确触发词，可包含多个；不从人物或服装介绍推断 Prompt" : "按作者实际标题审查 Civitai Detail、简介与触发词，保留可信的年龄、形态、服装及其他 Prompt 组"}</small></header>
            {loading && !metadata ? <div className="lora-detail-loading"><RefreshCw className="spin" size={22} />正在查询 Civitai 并解析本地模型...</div> : triggerGroups.length ? <div className="lora-trigger-groups">{triggerGroups.map((group, index) => {
              const value = group.words.join(", ");
              return <article key={`${group.label}-${index}`}><header><span>{String(index + 1).padStart(2, "0")}</span><strong>{group.label}</strong><CopyButton value={value} /></header><p>{value}</p><small>{sourceLabel(group.source)}</small></article>;
            })}</div> : <div className="lora-detail-empty"><FileText size={24} /><strong>无</strong><p>已审查 Civitai Detail、简介、触发词和本地模型元数据，没有发现可信 Prompt。</p></div>}
          </section>
          <section className="lora-detail-section lora-description-section">
            <header><div><span>PROMPT REVIEW</span><h3>自动审查摘要</h3></div></header>
            <p>{reviewSources.length ? `已检查 ${reviewSources.map((source) => source.label).join("、")}。${versionReviewText}${styleReview ? `保留 ${triggerGroups[0]?.words.length || 0} 个明确风格触发词` : `按作者标题保留 ${promptReview?.acceptedGroups || 0} 组有效 Prompt`}，另过滤 ${promptReview?.ignoredSegments || 0} 段公告、更新日志、模型推荐或非 Prompt 内容。` : found ? "已检查可用的 Civitai 信息，但没有发现可审查的简介、Detail 或触发词字段。" : "未获取到可审查的 Civitai Prompt 信息。"}</p>
          </section>
          {!styleReview && local.topTags?.length > 0 && <section className="lora-detail-section lora-training-tags">
            <header><div><span>TRAINING FREQUENCY</span><h3>训练高频标签</h3></div><CopyButton value={local.topTags.map((tag) => tag.word).join(", ")} /></header>
            <div>{local.topTags.map((tag) => <span key={tag.word}>{tag.word}<b>{tag.count}</b></span>)}</div>
            <p>这些词来自训练标签频率，不一定是必须触发词。</p>
          </section>}
        </main>
        <aside>
          <section className="lora-detail-section lora-metadata-section">
            <header><div><span>MODEL METADATA</span><h3>模型元数据</h3></div></header>
            {rows.length ? <dl>{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd title={String(value)}>{value}</dd><CopyButton value={String(value)} /></div>)}</dl> : <div className="lora-detail-empty compact"><strong>暂无可展示元数据</strong></div>}
          </section>
          {local.fields?.length > 0 && <details className="lora-raw-metadata"><summary>本地字段 · {local.fieldCount} 项</summary><div>{local.fields.map((field) => <p key={field.key}><span>{field.key}</span><b>{field.value}</b></p>)}</div></details>}
          {(error || metadata?.remoteError || metadata?.localMetadataError) && <p className="lora-detail-error">{error || [metadata?.remoteError && `Civitai 查询失败：${metadata.remoteError}`, metadata?.localMetadataError && `本地元数据解析失败：${metadata.localMetadataError}`].filter(Boolean).join("；")}</p>}
        </aside>
      </div>
    </section>
  </div>;
}
