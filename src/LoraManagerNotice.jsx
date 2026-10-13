import { CircleAlert, CircleCheck, Info, TriangleAlert, X } from "lucide-react";
import { loraNotice } from "./lora-notice.js";

// Only errors are alerts; everything else is a polite status so a success is never announced as a failure.
const ICONS = { error: CircleAlert, warning: TriangleAlert, success: CircleCheck, info: Info };

export default function LoraManagerNotice({ notice, onDismiss }) {
  const value = loraNotice(notice);
  if (!value) return null;
  const Icon = ICONS[value.kind];
  return <div className={`lora-manager-notice ${value.kind}`} role={value.kind === "error" ? "alert" : "status"}>
    <Icon size={16} aria-hidden="true" />
    <div className="lora-manager-notice-text">
      {value.title && <strong>{value.title}</strong>}
      {value.detail && <p>{value.detail}</p>}
      {value.diagnostics.length > 0 && <ul>{value.diagnostics.map((item, index) => <li key={`${item.name || ""}-${index}`}>{item.name && <b>{item.name}</b>}{item.message}</li>)}</ul>}
    </div>
    {onDismiss && <button type="button" aria-label="关闭提示" title="关闭提示" onClick={onDismiss}><X size={14} aria-hidden="true" /></button>}
  </div>;
}
