import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Download, RefreshCw } from "lucide-react";
import { formatFileSize } from "./format-size";

export default function TransparentBackgroundControl({
  enabled,
  disabled,
  modelId,
  models = [],
  runtimeAvailable,
  directory,
  downloadJob,
  onModelChange,
  onDownload,
  onToggle,
  className = "",
}) {
  const [open, setOpen] = useState(false);
  const pickerRef = useRef(null);
  const recommendedModels = models.filter((item) => !item.local);
  const localModels = models.filter((item) => item.local);
  const selectedModel = models.find((item) => item.id === modelId) || null;
  const selectedReady = Boolean(selectedModel?.installed && runtimeAvailable);
  const selectedDownload = downloadJob?.modelId === modelId ? downloadJob : null;
  const progress = selectedDownload?.totalBytes > 0
    ? Math.min(100, selectedDownload.currentBytes / selectedDownload.totalBytes * 100)
    : 0;

  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => {
      if (event.type === "keydown" && event.key !== "Escape") return;
      if (event.type === "pointerdown" && pickerRef.current?.contains(event.target)) return;
      setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [open]);

  return <div className={`transparent-tag-control ${enabled ? "active" : ""}${className ? ` ${className}` : ""}`}>
    <div className="transparent-model-picker" ref={pickerRef}>
      <button type="button" className="transparent-model-select" aria-haspopup="listbox" aria-expanded={open} disabled={disabled} onClick={() => setOpen((current) => !current)}>
        <span><b>{selectedModel?.label || "选择透明背景模型"}</b><small>{selectedModel ? `${selectedModel.installed ? "已安装" : formatFileSize(selectedModel.size)}${selectedModel.local ? " · 本地模型" : ""}` : "推荐模型与本地 ONNX"}</small></span>
        <ChevronDown size={14} />
      </button>
      {open && <div className="transparent-model-menu" role="listbox" aria-label="透明背景模型">
        <header><span>推荐模型</span><small>固定哈希 · 下载默认 8 路</small></header>
        {recommendedModels.map((item) => <div className={`transparent-model-option ${modelId === item.id ? "selected" : ""}`} role="option" aria-selected={modelId === item.id} key={item.id}>
          <button type="button" className="transparent-model-choice" onClick={() => { onModelChange(item.id); if (item.installed) setOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · {item.license || "本地使用"}</em></span></button>
          {item.installed ? <strong><Check size={12} />已安装</strong> : <button type="button" className="transparent-model-download" disabled={disabled || downloadJob?.active} onClick={() => onDownload(item.id)}>{downloadJob?.active && downloadJob.modelId === item.id ? <RefreshCw className="spin" size={12} /> : <Download size={12} />}{downloadJob?.active && downloadJob.modelId === item.id ? "下载中" : "下载"}</button>}
        </div>)}
        <header><span>本地模型</span><small title={directory}>放入 models/background-removal</small></header>
        {localModels.length ? localModels.map((item) => <div className={`transparent-model-option ${modelId === item.id ? "selected" : ""}`} role="option" aria-selected={modelId === item.id} key={item.id}><button type="button" className="transparent-model-choice" onClick={() => { onModelChange(item.id); setOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · 本地 ONNX</em></span></button><strong><Check size={12} />已发现</strong></div>) : <p>未发现额外 ONNX 模型</p>}
      </div>}
    </div>
    <div className={`transparent-mode-copy ${selectedDownload?.status || ""}`} aria-live="polite"><strong>透明背景模式</strong><span>{selectedDownload?.status === "error" ? selectedDownload.message : selectedDownload?.active ? selectedDownload.message : selectedReady ? `${selectedModel.label} · 已就绪` : selectedModel?.installed ? "模型已安装，等待 ONNX Runtime" : "选择模型后可在下拉栏内下载"}</span>{selectedDownload?.active && <><i><i style={{ width: `${progress}%` }} /></i><small>{formatFileSize(selectedDownload.currentBytes || 0)} / {formatFileSize(selectedDownload.totalBytes || selectedModel?.size || 0)}{selectedDownload.speedBps > 0 ? ` · ${formatFileSize(selectedDownload.speedBps)}/s` : ""} · {selectedDownload.route || "正在测速"} · {selectedDownload.connections || 8} 路</small></>}</div>
    <button type="button" className={`transparent-mode-switch ${enabled ? "active" : ""}`} role="switch" aria-checked={enabled} disabled={disabled} onClick={onToggle}><i /><span>{enabled ? "已启用" : "未启用"}</span></button>
  </div>;
}
