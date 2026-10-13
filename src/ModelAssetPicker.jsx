import { RefreshCw } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";

function itemOptions({ items, loading, value, missing, preserveUnavailable = false, missingOptionDisabled = false, emptyLabel, missingLabel, single = false }) {
  const unlisted = Boolean(value) && !items.some((item) => item.value === value);
  return [
    ...(loading ? [{ value: "", label: "正在加载模型目录...", disabled: true }] : []),
    ...(!loading && items.length === 0 ? [{ value: "", label: emptyLabel, disabled: true }] : []),
    ...((missing || preserveUnavailable) && unlisted ? [{ value, label: missingLabel || `${value}（文件已删除）`, ...(preserveUnavailable ? { disabled: true } : missingOptionDisabled ? { disabled: true } : {}) }] : []),
    ...items.map((item) => ({ value: item.value, label: item.name || item.label || item.value })),
  ];
}

function SplitAssetField({ field, loading, switching }) {
  const items = field.items || [];
  const missing = !loading && Boolean(field.value) && (field.missing ?? !items.some((item) => item.value === field.value));
  const pathText = field.error || (missing ? field.missingMessage || `${field.label}文件已删除，请选择其他资源` : field.directory || "正在解析模型路径...");
  return <label className="split-model-field" key={field.kind}>
    <span>{field.label}<small>{loading ? "扫描中" : `${items.length} 个`}</small></span>
    <span className="checkpoint-select"><WorkspaceSelect ariaLabel={field.ariaLabel || field.label} value={field.value ?? ""} disabled={switching || loading || items.length === 0} onChange={field.onChange} options={itemOptions({
      items, loading, value: field.value, missing, preserveUnavailable: field.preserveUnavailable, missingOptionDisabled: field.missingOptionDisabled, emptyLabel: "未检测到可用资源", missingLabel: field.missingLabel,
    })} /></span>
    <span className="checkpoint-path" title={field.error || (missing ? field.missingMessage || `${field.label}文件已删除，请选择其他资源` : field.directory)}>{pathText}</span>
  </label>;
}

/** Stateless model picker shared by 文生图 and XYZ; the owner still supplies its own draft handlers. */
export default function ModelAssetPicker({
  loading = false,
  refreshing = false,
  switching = false,
  error = "",
  fields = [],
  single = null,
  onRefresh,
  refreshDisabled = false,
  refreshTitle = "刷新模型组件",
  refreshAriaLabel = refreshTitle,
}) {
  const split = fields.length > 0;
  if (split) {
    const count = fields.reduce((total, field) => total + (field.items || []).length, 0);
    const hasMissing = fields.some((field) => !loading && field.value && (field.missing ?? !(field.items || []).some((item) => item.value === field.value)));
    return <div className={`checkpoint-picker split-model-picker${error || hasMissing ? " error" : ""}`}>
      <span className="checkpoint-label"><b>模型组件</b><span className="asset-list-actions"><small>{loading ? "扫描中" : `${count} 个资源`}</small><button type="button" className="asset-refresh" title={refreshTitle} aria-label={refreshAriaLabel} disabled={loading || refreshing || switching || refreshDisabled} onClick={(event) => { event.preventDefault(); onRefresh?.(); }}><RefreshCw className={refreshing ? "spin" : ""} size={13} /></button></span></span>
      {fields.map((field) => <SplitAssetField key={field.kind} field={field} loading={loading} switching={switching} />)}
    </div>;
  }
  const field = single || {};
  const items = field.items || [];
  const missing = !loading && Boolean(field.value) && (field.missing ?? !items.some((item) => item.value === field.value));
  const pathText = field.error || (missing ? field.missingMessage || "当前所选底模文件已删除，请选择其他模型" : field.directory || "正在解析模型路径...");
  return <div className={`checkpoint-picker${error || field.error || missing ? " error" : ""}`}>
    <span className="checkpoint-label"><b>底模选择</b><span className="asset-list-actions"><small>{loading ? "扫描中" : `${items.length} 个模型`}</small><button type="button" className="asset-refresh" title={refreshTitle} aria-label={refreshAriaLabel} disabled={loading || refreshing || switching || refreshDisabled} onClick={(event) => { event.preventDefault(); onRefresh?.(); }}><RefreshCw className={refreshing ? "spin" : ""} size={13} /></button></span></span>
    <span className="checkpoint-select"><WorkspaceSelect ariaLabel={field.ariaLabel || "底模选择"} value={field.value ?? ""} disabled={switching || loading || items.length === 0} onChange={field.onChange} options={itemOptions({
      items, loading, value: field.value, missing, preserveUnavailable: field.preserveUnavailable, missingOptionDisabled: field.missingOptionDisabled, emptyLabel: "未检测到可用底模", missingLabel: field.missingLabel, single: true,
    })} /></span>
    <span className="checkpoint-path" title={field.error || (missing ? field.missingMessage || "当前所选底模文件已删除，请选择其他模型" : field.directory)}>{pathText}</span>
  </div>;
}
