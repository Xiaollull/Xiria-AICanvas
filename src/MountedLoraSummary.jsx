import { Layers3, Square } from "lucide-react";
import { formatWeight } from "./lora-weight.js";

/** Compact mounted-list presentation shared by the Basic and XYZ sidebars. */
export default function MountedLoraSummary({ loras = [], presentationFor, bindHover, onOpenManager }) {
  if (loras.length === 0) return <button type="button" className="empty-lora" onClick={onOpenManager}><Layers3 size={16} /><span>尚未挂载 LoRA<small>点击管理，从分类库中选择</small></span></button>;
  return <div className="lora-summary">
    <div className="lora-summary-table">
      {loras.map((lora, index) => {
        const presentation = presentationFor?.(lora) || { title: lora.name || lora.value, coverUrl: lora.coverUrl || "" };
        const hover = bindHover?.(presentation) || {};
        return <div key={`${lora.value}:${index}`} className={`lora-summary-row ${lora.enabled === false ? "disabled" : ""} ${presentation.coverUrl ? "has-cover" : ""}`}
          title={presentation.rowTitle || `${lora.name || lora.value}${lora.category ? ` · 分类 ${lora.category}` : ""}`} {...hover}>
          <span className="lora-summary-name">{lora.enabled === false && <Square size={8} />}{presentation.title || lora.name || lora.value}</span>
          <b>{formatWeight(lora.weight)}</b>
        </div>;
      })}
    </div>
  </div>;
}
