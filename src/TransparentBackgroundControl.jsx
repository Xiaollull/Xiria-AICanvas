import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  const [menuStyle, setMenuStyle] = useState(null);
  const pickerRef = useRef(null);
  const menuRef = useRef(null);
  const recommendedModels = models.filter((item) => !item.local);
  const localModels = models.filter((item) => item.local);
  const selectedModel = models.find((item) => item.id === modelId) || null;
  const selectedReady = Boolean(selectedModel?.installed && runtimeAvailable);
  const selectedDownload = downloadJob?.modelId === modelId ? downloadJob : null;
  const progress = selectedDownload?.totalBytes > 0
    ? Math.min(100, selectedDownload.currentBytes / selectedDownload.totalBytes * 100)
    : 0;

  // The menu is portalled to the document, so a press inside it is no longer a press inside the
  // picker: it has to be excused explicitly or choosing a model would close the menu before the
  // click landed.
  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => {
      if (event.type === "keydown" && event.key !== "Escape") return;
      if (event.type === "pointerdown" && (pickerRef.current?.contains(event.target) || menuRef.current?.contains(event.target))) return;
      setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [open]);

  /**
   * Positioned from the trigger in viewport coordinates, the same way `WorkspaceSelect` is.
   *
   * `.transparent-tag-control` carries `z-index: 1`, deliberately, so it stays under the sticky
   * image-to-image run bar. That makes it a stacking context, and a stacking context is a ceiling:
   * the menu's own `z-index: 30` counted only against its siblings inside the control, never
   * against the page, so the prompt boxes and the size grid painted straight over the open menu.
   * Raising the number could not fix that from inside. Leaving the context does.
   */
  useEffect(() => {
    if (!open) return undefined;
    const positionMenu = () => {
      const rect = pickerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(610, window.innerWidth - 52);
      const spaceBelow = window.innerHeight - rect.bottom - 12;
      const spaceAbove = rect.top - 12;
      const placeAbove = spaceBelow < 220 && spaceAbove > spaceBelow;
      const maxHeight = Math.max(160, Math.min(520, window.innerHeight * 0.7, placeAbove ? spaceAbove : spaceBelow));
      const left = Math.min(Math.max(12, rect.left), Math.max(12, window.innerWidth - width - 12));
      setMenuStyle({ left, width, maxHeight, ...(placeAbove ? { bottom: window.innerHeight - rect.top + 6 } : { top: rect.bottom + 6 }) });
    };
    const onScroll = (event) => { if (!menuRef.current?.contains(event.target)) positionMenu(); };
    const frame = requestAnimationFrame(positionMenu);
    document.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", positionMenu);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", positionMenu);
    };
  }, [open]);

  return <div className={`transparent-tag-control ${enabled ? "active" : ""}${className ? ` ${className}` : ""}`}>
    <div className="transparent-model-picker" ref={pickerRef}>
      <button type="button" className="transparent-model-select" aria-haspopup="listbox" aria-expanded={open} disabled={disabled} onClick={() => setOpen((current) => !current)}>
        <span><b>{selectedModel?.label || "选择透明背景模型"}</b><small>{selectedModel ? `${selectedModel.installed ? "已安装" : formatFileSize(selectedModel.size)}${selectedModel.local ? " · 本地模型" : ""}` : "推荐模型与本地 ONNX"}</small></span>
        <ChevronDown size={14} />
      </button>
      {open && createPortal(<div ref={menuRef} className="transparent-model-menu" role="listbox" aria-label="透明背景模型" style={menuStyle || { visibility: "hidden" }}>
        <header><span>推荐模型</span><small>固定哈希 · 下载默认 8 路</small></header>
        {recommendedModels.map((item) => <div className={`transparent-model-option ${modelId === item.id ? "selected" : ""}`} role="option" aria-selected={modelId === item.id} key={item.id}>
          <button type="button" className="transparent-model-choice" onClick={() => { onModelChange(item.id); if (item.installed) setOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · {item.license || "本地使用"}</em></span></button>
          {item.installed ? <strong><Check size={12} />已安装</strong> : <button type="button" className="transparent-model-download" disabled={disabled || downloadJob?.active} onClick={() => onDownload(item.id)}>{downloadJob?.active && downloadJob.modelId === item.id ? <RefreshCw className="spin" size={12} /> : <Download size={12} />}{downloadJob?.active && downloadJob.modelId === item.id ? "下载中" : "下载"}</button>}
        </div>)}
        <header><span>本地模型</span><small title={directory}>放入 models/background-removal</small></header>
        {localModels.length ? localModels.map((item) => <div className={`transparent-model-option ${modelId === item.id ? "selected" : ""}`} role="option" aria-selected={modelId === item.id} key={item.id}><button type="button" className="transparent-model-choice" onClick={() => { onModelChange(item.id); setOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · 本地 ONNX</em></span></button><strong><Check size={12} />已发现</strong></div>) : <p>未发现额外 ONNX 模型</p>}
      </div>, document.body)}
    </div>
    <div className={`transparent-mode-copy ${selectedDownload?.status || ""}`} aria-live="polite"><strong>透明背景模式</strong><span>{selectedDownload?.status === "error" ? selectedDownload.message : selectedDownload?.active ? selectedDownload.message : selectedReady ? `${selectedModel.label} · 已就绪` : selectedModel?.installed ? "模型已安装，等待 ONNX Runtime" : "选择模型后可在下拉栏内下载"}</span>{selectedDownload?.active && <><i><i style={{ width: `${progress}%` }} /></i><small>{formatFileSize(selectedDownload.currentBytes || 0)} / {formatFileSize(selectedDownload.totalBytes || selectedModel?.size || 0)}{selectedDownload.speedBps > 0 ? ` · ${formatFileSize(selectedDownload.speedBps)}/s` : ""} · {selectedDownload.route || "正在测速"} · {selectedDownload.connections || 8} 路</small></>}</div>
    <button type="button" className={`transparent-mode-switch ${enabled ? "active" : ""}`} role="switch" aria-checked={enabled} disabled={disabled} onClick={onToggle}><i /><span>{enabled ? "已启用" : "未启用"}</span></button>
  </div>;
}
