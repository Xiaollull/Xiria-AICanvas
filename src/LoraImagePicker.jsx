import { useEffect, useRef, useState } from "react";
import { FolderOpen, Images, RefreshCw, Upload, X } from "lucide-react";

import { readImageFileAsDataUrl } from "./lora-card-image.js";

// Where a card picture comes from: something the user generated here, or a file
// they already have.
//
// The generated side reads the same output history the gallery does, so a cover
// can be the picture that actually came out of the combination rather than a
// screenshot of it. The session listing is what the backend answers with by
// default; the folder list is offered as well because a fresh start has no
// session images and the user's earlier work is still on disk.

const SESSION_FOLDER = "";

export default function LoraImagePicker({ title = "选择图片", busy = false, onPick, onClose }) {
  const dialogRef = useRef(null);
  const fileRef = useRef(null);
  const closeRef = useRef(onClose);
  const [folder, setFolder] = useState(SESSION_FOLDER);
  const [folders, setFolders] = useState([]);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  closeRef.current = onClose;

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopImmediatePropagation();
      closeRef.current();
    };
    window.addEventListener("keydown", onKeyDown, true);
    const frame = window.requestAnimationFrame(() => dialogRef.current?.querySelector("button")?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKeyDown, true);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(`/api/inference/history${folder ? `?folder=${encodeURIComponent(folder)}` : ""}`, { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "无法读取生成记录");
        return payload;
      })
      .then((payload) => {
        if (cancelled) return;
        setFolders(Array.isArray(payload.folders) ? payload.folders : []);
        setFiles((Array.isArray(payload.cards) ? payload.cards : []).flatMap((card) => Array.isArray(card.files) ? card.files : []));
      })
      .catch((loadError) => !cancelled && setError(loadError.message))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [folder]);

  const pickUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      onPick(await readImageFileAsDataUrl(file));
    } catch (readError) {
      setError(readError.message);
    }
  };

  return <div className="lora-picker-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="lora-picker" role="dialog" aria-modal="true" aria-label={title}>
      <header>
        <div><span>CARD ARTWORK</span><h3>{title}</h3></div>
        <div>
          <button type="button" disabled={busy} onClick={() => fileRef.current?.click()}><Upload size={13} />上传本地图片</button>
          <button type="button" aria-label="关闭图片选择" onClick={onClose}><X size={17} /></button>
        </div>
      </header>
      <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={pickUpload} />
      <nav className="lora-picker-folders" aria-label="生成记录目录">
        <button type="button" className={folder === SESSION_FOLDER ? "active" : ""} onClick={() => setFolder(SESSION_FOLDER)}><Images size={13} />本次会话</button>
        {folders.map((entry) => (
          <button type="button" key={entry.id} className={folder === entry.id ? "active" : ""} title={entry.label} onClick={() => setFolder(entry.id)}>
            <FolderOpen size={13} />{entry.name}<b>{entry.count}</b>
          </button>
        ))}
      </nav>
      {error && <p className="lora-picker-error">{error}</p>}
      {loading
        ? <div className="lora-picker-loading"><RefreshCw className="spin" size={22} />正在读取生成记录</div>
        : files.length
          ? <div className="lora-picker-grid">
            {files.map((file) => (
              <button type="button" key={file.id} disabled={busy} title={file.name} onClick={() => onPick(file.url)}>
                <img src={file.url} alt="" loading="lazy" />
              </button>
            ))}
          </div>
          : <div className="lora-picker-empty">
            <Images size={26} />
            <strong>{folder ? "该目录没有图片" : "本次会话还没有生成图片"}</strong>
            <p>换一个输出目录，或直接上传一张本地图片。</p>
          </div>}
      <footer><small>图片会缩放到 1024 像素以内另存一份，原图不受影响。</small>{busy && <span><RefreshCw className="spin" size={12} />正在保存</span>}</footer>
    </section>
  </div>;
}
