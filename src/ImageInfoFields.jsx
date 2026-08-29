import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, Copy, Share2 } from "lucide-react";
import { groupParameters } from "./metadata-explorer";

// The two presentational pieces the facts panel and the details dialog both use.
// They live here so neither file has to import the other.

// Where a model was found decides how its name reads. "Missing" is the only
// one the user has to act on, so it is the only one that shouts.
export function ModelName({ match, fallback = "", className = "" }) {
  const name = match?.name || fallback;
  if (!name) return <span className="model-name-unset">未记录</span>;
  if (!match || match.status === "unknown") return <span className={className}>{name}</span>;
  if (match.status === "missing") {
    return <span className={`model-name-missing ${className}`} title="本地模型目录与共享目录中都没有这个模型">
      <AlertTriangle size={12} />{name}
    </span>;
  }
  return <span className={`model-name-found ${match.status} ${className}`}>
    {name}
    <i title={match.match?.label || ""}>{match.status === "shared" ? <><Share2 size={10} />共享</> : <><Check size={10} />本地</>}</i>
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
      {value ? <button type="button" onClick={() => { void navigator.clipboard?.writeText(value).then(() => setCopied(true)).catch(() => {}); }}>
        {copied ? <Check size={12} /> : <Copy size={12} />}{copied ? "已复制" : "复制"}
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
  return <button type="button" className="metadata-copy" onClick={() => { void navigator.clipboard?.writeText(String(value ?? "")).then(() => setCopied(true)).catch(() => {}); }}>
    {copied ? <Check size={size} /> : <Copy size={size} />}{copied ? "已复制" : label}
  </button>;
}
