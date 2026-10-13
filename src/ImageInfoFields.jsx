import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, Copy, Share2 } from "lucide-react";
import { copyText } from "./clipboard";
import { groupParameters } from "./metadata-explorer";

// The two presentational pieces the facts panel and the details dialog both use.
// They live here so neither file has to import the other.

// Where a model was found decides how its name reads. "Missing" and "ambiguous"
// are the two the user has to act on, so they are the ones that shout.
//
// A resolved model is shown by its last folder and real filename
// (`Style_九V6.0\matcha.safetensors`), with the name the picture recorded kept
// beside it when the two differ — the label changes, the mounted ID does not.
export function ModelName({ match, fallback = "", className = "", lora = false }) {
  const name = match?.name || fallback;
  if (!name) return <span className="model-name-unset">未记录</span>;
  if (!match || match.status === "unknown") return <span className={className}>{name}</span>;
  if (match.status === "missing") {
    // A LoRA with no file of its name here has nothing to select: it reads as 无选择, with the
    // recorded name beside it, rather than as a warning on every row of a borrowed picture.
    if (lora) {
      return <span className={`model-name-unselected ${className}`} title={`本地模型目录与共享目录中都没有这个 LoRA：${name}`}>
        无选择<small className="model-name-recorded">{name}</small>{match.explanation && <small className="model-name-relink">{match.explanation}</small>}
      </span>;
    }
    return <span className={`model-name-missing ${className}`} title="本地模型目录与共享目录中都没有这个模型">
      <AlertTriangle size={12} />{name}
    </span>;
  }
  if (match.status === "ambiguous") {
    const candidates = (match.candidates || []).map((item) => `${item.display || item.value || item.name}${item.engine ? ` · ${item.engine}` : ""}${item.origin === "shared" ? " · 共享" : ""}`);
    const more = (match.count || candidates.length) > candidates.length ? `\n另有 ${(match.count || 0) - candidates.length} 个` : "";
    return <span className={`model-name-ambiguous ${className}`} title={match.explanation || `有歧义：本地有 ${match.count || candidates.length} 个同名文件，无法确定是哪一个${match.hashChecked ? "（记录的哈希也无法区分）" : ""}\n${candidates.join("\n")}${more}`}>
      <AlertTriangle size={12} />{name}<i>有歧义 · {match.count || candidates.length} 个候选</i>{match.explanation && <small className="model-name-relink">{match.explanation}</small>}
      {match.auditKind === "path-reuse" && candidates.map((candidate, index) => <small className="model-name-relink" key={index}>候选：{candidate}</small>)}
    </span>;
  }
  const display = match.relink?.currentValue || match.display || match.match?.display || name;
  const detail = [
    match.match?.label || "",
    match.precision ? `匹配方式 ${match.precision}` : "",
    match.duplicates ? `另有 ${match.duplicates} 个内容相同的副本` : "",
  ].filter(Boolean).join(" · ");
  // Chosen among same-named files by where the picture's other models live — said, not hidden.
  const choice = match.chosenBy
    ? {
      label: `${match.chosenBy === "folder" ? "同目录" : "就近"} · ${match.count} 个同名`,
      title: `${match.count} 个同名文件，${match.chosenBy === "folder" ? "按图片中其他模型所在的文件夹选择了这一个" : "图片其他模型所在的文件夹里没有，选择了位置最接近的一个"}${match.hashMismatch ? "；图片记录的哈希与这些文件都不一致" : ""}`,
    }
    : null;
  return <span className={`model-name-found ${match.status} ${className}`} title={display !== name ? `图片记录：${name}` : undefined}>
    {display}
    {display !== name && <small className="model-name-recorded">{name}</small>}
    {match.relink?.kind === "relocate" && <small className="model-name-relink">原位置 → 当前位置：{match.relink.originalValue || name} → {match.relink.currentValue}</small>}
    {match.relink?.kind === "replace" && <small className="model-name-relink">原文件缺失，使用指定替代：{match.relink.originalValue || name} → {match.relink.currentValue}</small>}
    {choice && <em className="model-name-choice" title={choice.title}>{choice.label}</em>}
    <i title={detail}>{match.status === "shared" ? <><Share2 size={10} />共享</> : <><Check size={10} />本地</>}</i>
  </span>;
}

export function CopyField({ label, value, empty }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = window.setTimeout(() => setCopied(false), 1400);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return <div className="info-prompt-field">
    <header>
      <span>{label}</span>
      {value ? <button type="button" onClick={() => { void copyText(value).then((ok) => setCopied(ok ? "ok" : "failed")); }}>
        {copied === "ok" ? <Check size={12} /> : <Copy size={12} />}{copied === "ok" ? "已复制" : copied === "failed" ? "复制失败" : "复制"}
      </button> : null}
    </header>
    <p className={value ? "" : "empty"}>{value || empty}</p>
  </div>;
}

function ParameterCells({ items }) {
  return <div className="info-parameter-grid">
    {items.map((item, position) => <div key={`${item.label}-${position}`}>
      <span>{item.label}</span>
      <b>{item.value}</b>
    </div>)}
  </div>;
}

// One block per module the producer described. A record with nothing to group —
// every ComfyUI and XiriaCanvas image — keeps the plain grid, because a single
// heading over a single grid is a line that says nothing.
export function ParameterModules({ parameters }) {
  const groups = useMemo(() => groupParameters(parameters), [parameters]);
  if (!groups.length) return null;
  if (groups.length === 1) return <ParameterCells items={groups[0].items} />;
  return <div className="info-parameter-modules">
    {groups.map((group) => <section key={group.key}>
      <h4>{group.label}<b>{group.items.length}</b></h4>
      <ParameterCells items={group.items} />
    </section>)}
  </div>;
}

// Copies any string and says so for a moment. Used wherever a block has more
// content than a field label can carry.
export function CopyButton({ value, label = "复制", size = 11 }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = window.setTimeout(() => setCopied(false), 1400);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return <button type="button" className="metadata-copy" onClick={() => { void copyText(String(value ?? "")).then((ok) => setCopied(ok ? "ok" : "failed")); }}>
    {copied === "ok" ? <Check size={size} /> : <Copy size={size} />}{copied === "ok" ? "已复制" : copied === "failed" ? "复制失败" : label}
  </button>;
}
