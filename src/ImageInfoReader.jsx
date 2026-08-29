import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  FolderSearch,
  Maximize2,
  RefreshCw,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import { formatFileSize } from "./format-size";
import { hostPathExample } from "./host-platform";
import ImageInfoDetails from "./ImageInfoDetails";
import ImageInfoApply from "./ImageInfoApply";
import { CopyField, ModelName, ParameterModules } from "./ImageInfoFields";
import { imageInfoApplyFields } from "./image-info-apply";

const DIRECTORY_PLACEHOLDER = `批量读取：输入图片文件夹绝对路径，例如 ${hostPathExample("imageDirectory")}`;
const MAX_UPLOAD_BYTES = 64 * 1024 * 1024;
const IMAGE_FILE_NAME = /\.(?:png|jpe?g|webp)$/i;

function isImageFile(file) {
  return Boolean(file) && (file.type?.startsWith("image/") || IMAGE_FILE_NAME.test(file.name || ""));
}

export default function ImageInfoReader({ onApplyParameters }) {
  const [mode, setMode] = useState("single");
  const [directory, setDirectory] = useState("");
  const [files, setFiles] = useState([]);
  const [index, setIndex] = useState(0);
  const [scannedDirectory, setScannedDirectory] = useState("");
  const [truncated, setTruncated] = useState(false);
  const [info, setInfo] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [applyOpen, setApplyOpen] = useState(false);
  const objectUrlRef = useRef("");
  const requestRef = useRef(0);
  const dragDepthRef = useRef(0);
  const uploadInputRef = useRef(null);

  const releaseObjectUrl = () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = "";
  };
  useEffect(() => releaseObjectUrl, []);

  const readUploadedFile = async (file) => {
    if (!file) return;
    if (!isImageFile(file)) {
      setError("请使用 PNG、JPEG 或 WebP 图片");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`图片超过 ${formatFileSize(MAX_UPLOAD_BYTES)} 上限`);
      return;
    }
    const token = ++requestRef.current;
    setLoading(true);
    setError("");
    setMode("single");
    setFiles([]);
    releaseObjectUrl();
    // The picture is shown straight from the browser's own copy; only the
    // bytes needed for parsing go to the control plane.
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("无法读取所选文件"));
        reader.readAsDataURL(file);
      });
      const response = await fetch("/api/image-info/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl, name: file.name }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法解析图片信息");
      if (token === requestRef.current) setInfo(payload.info);
    } catch (readError) {
      if (token === requestRef.current) { setInfo(null); setError(readError.message); }
    } finally {
      if (token === requestRef.current) setLoading(false);
    }
  };

  const readFromDirectory = async (targetDirectory, name) => {
    const token = ++requestRef.current;
    setLoading(true);
    setError("");
    releaseObjectUrl();
    setImageUrl(`/api/image-info/preview?directory=${encodeURIComponent(targetDirectory)}&name=${encodeURIComponent(name)}`);
    try {
      const response = await fetch("/api/image-info/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ directory: targetDirectory, name }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法解析图片信息");
      if (token === requestRef.current) setInfo(payload.info);
    } catch (readError) {
      if (token === requestRef.current) { setInfo(null); setError(readError.message); }
    } finally {
      if (token === requestRef.current) setLoading(false);
    }
  };

  const scanDirectory = async () => {
    const target = directory.trim();
    if (!target || loading) return;
    setLoading(true);
    setError("");
    setInfo(null);
    releaseObjectUrl();
    setImageUrl("");
    try {
      const response = await fetch("/api/image-info/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ directory: target }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取目录");
      setMode("batch");
      setFiles(payload.files || []);
      setScannedDirectory(payload.directory);
      setTruncated(payload.truncated === true);
      setIndex(0);
      if (!payload.files?.length) {
        setError("该目录下没有图片文件");
        setLoading(false);
        return;
      }
      await readFromDirectory(payload.directory, payload.files[0].name);
    } catch (scanError) {
      setError(scanError.message);
      setLoading(false);
    }
  };

  const goTo = (nextIndex) => {
    if (mode !== "batch" || !files.length) return;
    const bounded = Math.min(files.length - 1, Math.max(0, nextIndex));
    if (bounded === index && info) return;
    setIndex(bounded);
    void readFromDirectory(scannedDirectory, files[bounded].name);
  };

  const handleDragEnter = (event) => {
    event.preventDefault();
    if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
    dragDepthRef.current += 1;
    setDragActive(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (!dragDepthRef.current) setDragActive(false);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setDragActive(false);
    const file = Array.from(event.dataTransfer?.files || []).find(isImageFile);
    if (!file) {
      setError("请拖入 PNG、JPEG 或 WebP 图片");
      return;
    }
    void readUploadedFile(file);
  };

  const handlePaste = (event) => {
    const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.kind === "file" && entry.type.startsWith("image/"));
    const file = item?.getAsFile();
    if (!file) return;
    event.preventDefault();
    void readUploadedFile(file);
  };

  useEffect(() => {
    if (mode !== "batch" || files.length < 2) return undefined;
    const onKey = (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === "ArrowLeft") { event.preventDefault(); goTo(index - 1); }
      if (event.key === "ArrowRight") { event.preventDefault(); goTo(index + 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, files, index, scannedDirectory]);

  const applicableFields = imageInfoApplyFields(info);
  const missingCount = [
    info?.checkpointMatch?.status === "missing" ? 1 : 0,
    (info?.loras || []).filter((item) => item.match?.status === "missing").length,
  ].reduce((total, value) => total + value, 0);

  return <><section className="image-info-tool">
    <header className="image-info-head">
      <div>
        <span className="eyebrow">IMAGE METADATA READER</span>
        <p>读取 ComfyUI、SD-WebUI 与本程序输出的 PNG 生成信息。原图被再次编辑或压缩后信息会丢失，此时显示「无元数据」。</p>
      </div>
      <div className="image-info-sources"><span>ComfyUI</span><span>SD-WebUI</span><span>XiriaCanvas</span><span>NovelAI</span></div>
    </header>

    <div className="image-info-inputs">
      <label className="image-info-upload">
        <input ref={uploadInputRef} type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { void readUploadedFile(event.target.files?.[0]); event.target.value = ""; }} />
        <Upload size={15} /><span><strong>选择单张图片</strong><small>拖入预览区，或点击后粘贴截图</small></span>
      </label>
      <div className="image-info-directory">
        <FolderSearch size={15} />
        <input
          value={directory}
          spellCheck="false"
          placeholder={DIRECTORY_PLACEHOLDER}
          onChange={(event) => setDirectory(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void scanDirectory(); } }}
        />
        <button type="button" disabled={!directory.trim() || loading} onClick={() => void scanDirectory()}>
          {loading && mode === "batch" && !files.length ? <RefreshCw className="spin" size={13} /> : <FolderSearch size={13} />}批量读取
        </button>
      </div>
    </div>

    {error && <p className="image-info-error"><X size={13} />{error}</p>}

    {mode === "batch" && files.length > 0 && <div className="image-info-pager">
      <button type="button" disabled={index === 0 || loading} onClick={() => goTo(index - 1)}><ChevronLeft size={14} />上一张</button>
      <div>
        <strong>{files[index]?.name}</strong>
        <span>{index + 1} / {files.length}{truncated ? " · 已达到目录读取上限" : ""} · ← → 切换</span>
      </div>
      <button type="button" disabled={index >= files.length - 1 || loading} onClick={() => goTo(index + 1)}>下一张<ChevronRight size={14} /></button>
    </div>}

    <div className="image-info-body">
      <div
        className={`image-info-canvas ${!imageUrl ? "empty" : ""} ${dragActive ? "drag-active" : ""}`}
        role="region"
        tabIndex={0}
        aria-label="图片预览区，支持拖拽或粘贴图片"
        onClick={(event) => { event.currentTarget.focus(); if (!imageUrl) uploadInputRef.current?.click(); }}
        onKeyDown={(event) => { if (!imageUrl && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); uploadInputRef.current?.click(); } }}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onPaste={handlePaste}
      >
        {imageUrl ? <div className="image-info-image-stage"><img src={imageUrl} alt={info?.name || "预览"} /></div> : <div className="image-info-empty-dropzone">
          <Upload size={30} />
          <strong>拖入、粘贴或点击选择图片</strong>
          <span>PNG · JPEG · WebP</span>
        </div>}
        {dragActive && <div className="image-info-drop-overlay"><Upload size={27} /><strong>松开以读取图片</strong><span>图片会从浏览器本地读取并解析</span></div>}
        {loading && <div className="image-info-loading"><RefreshCw className="spin" size={20} /></div>}
      </div>

      <aside className="image-info-panel">
        <header className="image-info-panel-head">
          <span>解析结果</span>
          {onApplyParameters && <button type="button" className="image-info-apply-open" disabled={!applicableFields.length} onClick={() => setApplyOpen((current) => !current)}><WandSparkles size={12} />应用参数</button>}
          <button type="button" disabled={!info} onClick={() => setDetailsOpen(true)}><Maximize2 size={12} />详细信息</button>
        </header>
        {!info ? <div className="image-info-empty"><span>读取到的生成信息会显示在这里</span></div>
          : info.status === "empty" ? <div className="image-info-none">
            <AlertTriangle size={26} />
            <strong>无元数据</strong>
            <span>这张图片没有携带生成信息。通常是因为它经过二次编辑、截图、转码或由不写入元数据的工具导出。</span>
          </div>
            : info.status === "unsupported" ? <div className="image-info-none">
              <AlertTriangle size={26} />
              <strong>非 PNG 文件</strong>
              <span>当前只解析 PNG 文本块。该文件可以预览，但其中的生成信息（若存在）保存在本工具尚未读取的格式中。</span>
            </div> : <>
              <div className="info-section info-identity">
                <div><span>来源</span><strong>{info.sourceLabel}</strong></div>
                {info.engine && <div><span>引擎</span><strong>{info.engine}</strong></div>}
                <div className="info-identity-model">
                  <span>底模</span>
                  <ModelName match={info.checkpointMatch} fallback={info.checkpoint} className="info-identity-model-name" />
                </div>
                {info.checkpointHash && <div><span>模型哈希</span><code>{info.checkpointHash.slice(0, 16)}</code></div>}
              </div>

              {applyOpen && onApplyParameters && <ImageInfoApply info={info} onApply={onApplyParameters} onClose={() => setApplyOpen(false)} />}

              {missingCount > 0 && <p className="info-missing-warning">
                <AlertTriangle size={13} />
                有 {missingCount} 个模型在本地模型目录和共享目录中都找不到，按此信息复现会失败。
              </p>}

              {info.animaAssets && <div className="info-section">
                <h3>Anima 组件</h3>
                <div className="info-asset-row"><span>文本编码器</span><ModelName match={info.animaAssets.text_encoder.match} fallback={info.animaAssets.text_encoder.name} /></div>
                <div className="info-asset-row"><span>VAE</span><ModelName match={info.animaAssets.vae.match} fallback={info.animaAssets.vae.name} /></div>
              </div>}

              <div className="info-section">
                <CopyField label="正向提示词" value={info.positive} empty="未记录正向提示词" />
                <CopyField label="负向提示词" value={info.negative} empty="未记录负向提示词" />
              </div>

              {info.parameters?.length > 0 && <div className="info-section">
                <h3>生成参数</h3>
                <ParameterModules parameters={info.parameters} />
              </div>}

              <div className="info-section">
                <h3>LoRA<b>{info.loras?.length || 0}</b></h3>
                {info.loras?.length ? <div className="info-lora-list">
                  {info.loras.map((item, position) => <div key={`${item.name}-${position}`}>
                    <ModelName match={item.match} fallback={item.name} />
                    <b>{item.weight === null || item.weight === undefined || Number.isNaN(item.weight) ? "权重未记录" : item.weight}</b>
                  </div>)}
                </div> : <p className="info-section-empty">未使用 LoRA</p>}
              </div>

              {info.flags?.length > 0 && <div className="info-section">
                <h3>功能标记</h3>
                <div className="info-flag-list">
                  {info.flags.map((flag) => <span key={flag.id} className={flag.enabled ? "on" : ""}>
                    {flag.enabled ? <Check size={11} /> : null}{flag.label}{flag.enabled && flag.detail ? <i>{flag.detail}</i> : null}
                  </span>)}
                </div>
              </div>}
            </>}
      </aside>
    </div>
  </section>
  {detailsOpen && info && <ImageInfoDetails info={info} onClose={() => setDetailsOpen(false)} />}
  </>;
}
