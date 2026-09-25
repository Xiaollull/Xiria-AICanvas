import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Check, ChevronRight, Folder, ImageIcon, RefreshCw, X } from "lucide-react";
import { useDialogLifecycle } from "./gallery-core";

// Choosing pictures for a curated card from the output folder.
//
// Everything goes through the history API the output browser already uses: folders are addressed
// by the opaque tokens `/api/inference/history` hands out, and a picture by its `asset_id`. The
// browser never names a file on disk; saving the card sends the `asset_id` and the server copies
// the file itself (`GalleryImageInput.asset_id`). A folder can hold thousands of full-size PNGs, so
// the grid shows a bounded page at a time and each thumbnail is decoded only once it scrolls in.

export const OUTPUT_DRAWER_PAGE = 24;

async function readHistory(folderId) {
  const query = folderId ? `?folder=${encodeURIComponent(folderId)}` : "";
  const response = await fetch(`/api/inference/history${query}`, { cache: "no-store" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || "无法读取输出目录");
  return payload;
}

export default function GalleryOutputDrawer({ onClose, onPick, maxSelectable = 32 }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  // `null` until the root token is known; the root listing without a token is the session view,
  // which is not what "browse the output folder" means.
  const [folderId, setFolderId] = useState(null);
  const [listing, setListing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [limit, setLimit] = useState(OUTPUT_DRAWER_PAGE);
  const [selected, setSelected] = useState(() => new Map());
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    (async () => {
      let token = folderId;
      if (token === null) token = (await readHistory("")).directory?.id || "";
      const payload = await readHistory(token);
      if (cancelled) return;
      if (folderId === null) setFolderId(token);
      setListing(payload);
      setLimit(OUTPUT_DRAWER_PAGE);
    })().catch((readError) => { if (!cancelled) setError(readError.message); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [folderId, reload]);

  const directory = listing?.directory || null;
  const files = useMemo(() => (listing?.cards || []).flatMap((card) => card.files || []), [listing]);
  const visible = files.slice(0, limit);
  const toggle = (file) => setSelected((current) => {
    const next = new Map(current);
    if (next.has(file.id)) next.delete(file.id);
    else if (next.size < maxSelectable) next.set(file.id, file);
    return next;
  });

  return <div className="gallery-output-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <aside ref={dialogRef} className="gallery-output-drawer" role="dialog" aria-modal="true" aria-labelledby="gallery-output-title" tabIndex="-1">
      <header>
        <div><span>OUTPUT</span><h3 id="gallery-output-title">从 output 选择</h3><p>{directory ? directory.label : "正在读取…"}</p></div>
        <button type="button" aria-label="关闭输出目录" onClick={onClose}><X size={18} /></button>
      </header>
      <nav className="gallery-output-nav" aria-label="输出文件夹">
        <button type="button" disabled={!directory?.parent_id || loading} onClick={() => setFolderId(directory.parent_id)}><ArrowLeft size={13} />上一级</button>
        <span title={directory?.label || ""}>{directory?.label || "/"}</span>
        <button type="button" disabled={loading} aria-label="刷新输出目录" onClick={() => setReload((current) => current + 1)}><RefreshCw className={loading ? "spin" : ""} size={13} /></button>
      </nav>
      <div className="gallery-output-body">
        {error && <p className="gallery-form-error">{error}</p>}
        {directory?.folders?.length > 0 && <div className="gallery-output-folders">
          {directory.folders.map((folder) => <button type="button" key={folder.id} onClick={() => setFolderId(folder.id)}>
            <Folder size={15} /><span><strong>{folder.name}</strong><small>{folder.count} 张{folder.folder_count ? ` · ${folder.folder_count} 个文件夹` : ""}</small></span><ChevronRight size={13} />
          </button>)}
        </div>}
        {!loading && !error && !files.length && <div className="gallery-output-empty"><ImageIcon size={26} /><span>这个文件夹里没有图片</span></div>}
        <div className="gallery-output-grid">
          {visible.map((file) => {
            const chosen = selected.has(file.id);
            return <button type="button" key={file.id} className={chosen ? "selected" : ""} aria-pressed={chosen} title={file.name} onClick={() => toggle(file)}>
              <img src={file.url} alt="" loading="lazy" decoding="async" />
              <span>{file.name}</span>
              {chosen && <i><Check size={12} /></i>}
            </button>;
          })}
        </div>
        {files.length > visible.length && <button type="button" className="gallery-output-more" onClick={() => setLimit((current) => current + OUTPUT_DRAWER_PAGE)}>再显示 {Math.min(OUTPUT_DRAWER_PAGE, files.length - visible.length)} 张（共 {files.length} 张）</button>}
      </div>
      <footer>
        <span>{selected.size ? `已选 ${selected.size} 张` : "点击图片选择，可多选"}</span>
        <button type="button" onClick={onClose}>取消</button>
        <button type="button" className="primary" disabled={!selected.size} onClick={() => onPick([...selected.values()])}>加入卡片</button>
      </footer>
    </aside>
  </div>;
}
