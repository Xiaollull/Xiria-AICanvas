import { useRef, useState } from "react";
import { Check, Download, FileArchive, FolderOutput, RefreshCw, Upload, X } from "lucide-react";
import { galleryRequest, useDialogLifecycle } from "./gallery-core";
import { cleanZipPath, formatBytes, isZipFile, uploadPackage, zipPathIssue } from "./gallery-transfer.js";

function startDownload(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.rel = "noopener";
  document.body.append(link);
  link.click();
  link.remove();
}

function outputPath(root, folder) {
  const base = String(root || "outputs").replace(/[\\/]+$/, "");
  return `${base}/${folder}`;
}

export default function GalleryTransferDialog({ collections, cards, currentCollection = null, onClose, onImported }) {
  const [tab, setTab] = useState("export");
  const [selected, setSelected] = useState(() => new Set(currentCollection ? [currentCollection] : []));
  const [download, setDownload] = useState(true);
  const [zipPath, setZipPath] = useState("");
  const [file, setFile] = useState(null);
  const [folder, setFolder] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [receipt, setReceipt] = useState(null);
  const dragDepth = useRef(0);
  const fileInput = useRef(null);
  const dialogRef = useDialogLifecycle(true, onClose, "", !busy);

  const imageCount = (id) => cards.filter((card) => card.collection_id === id).reduce((total, card) => total + (card.images?.length || 0), 0);
  const chosen = collections.filter((collection) => selected.has(collection.id));
  const chosenCards = chosen.reduce((total, collection) => total + (collection.card_count || 0), 0);
  const chosenImages = chosen.reduce((total, collection) => total + imageCount(collection.id), 0);
  const pathIssue = zipPathIssue(zipPath);
  const canImport = Boolean(file) || (Boolean(cleanZipPath(zipPath)) && !pathIssue);

  const switchTab = (next) => {
    if (busy) return;
    setTab(next);
    setError("");
    setReceipt(null);
  };
  const toggle = (id) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const chooseFile = (candidate) => {
    if (!candidate) return;
    if (!isZipFile(candidate)) {
      setError(`“${candidate.name}”不是 .zip 文件`);
      return;
    }
    setError("");
    setFile(candidate);
  };
  const acceptsFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  const dragEnter = (event) => {
    if (!acceptsFiles(event) || busy) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };
  const dragOver = (event) => {
    if (!acceptsFiles(event) || busy) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };
  const dragLeave = (event) => {
    if (!acceptsFiles(event)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (!dragDepth.current) setDragging(false);
  };
  const drop = (event) => {
    if (!acceptsFiles(event) || busy) return;
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const files = [...(event.dataTransfer.files || [])];
    if (files.length > 1) {
      setError("一次只能导入一个 ZIP");
      return;
    }
    chooseFile(files[0]);
  };

  const runExport = async () => {
    if (!chosen.length || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await galleryRequest("/transfer/export", { method: "POST", body: JSON.stringify({ collection_ids: chosen.map((collection) => collection.id) }) });
      if (download) startDownload(result.zip_url, `${result.folder}.zip`);
      setReceipt({ kind: "export", downloaded: download, ...result });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  const runImport = async () => {
    if (!canImport || busy) return;
    setBusy(true);
    setError("");
    try {
      const target = folder.trim() || null;
      const result = file
        ? await uploadPackage(file, target)
        : await galleryRequest("/transfer/import", { method: "POST", body: JSON.stringify({ zip_path: cleanZipPath(zipPath), folder: target }) });
      setReceipt({ kind: "import", ...result });
      setFile(null);
      setZipPath("");
      setFolder("");
      await onImported?.(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  const exportBody = <>
    <div className="gallery-transfer-toolbar">
      <span>选择要导出的收藏夹</span>
      <div>
        <button type="button" disabled={busy || !collections.length} onClick={() => setSelected(new Set(collections.map((collection) => collection.id)))}>全选</button>
        <button type="button" disabled={busy || !selected.size} onClick={() => setSelected(new Set())}>取消选择</button>
      </div>
    </div>
    {collections.length ? <ul className="gallery-transfer-list" aria-label="收藏夹">
      {collections.map((collection) => <li key={collection.id}><label className={selected.has(collection.id) ? "selected" : ""}>
        <input type="checkbox" checked={selected.has(collection.id)} disabled={busy} onChange={() => toggle(collection.id)} />
        <i aria-hidden="true">{selected.has(collection.id) && <Check size={12} strokeWidth={3} />}</i>
        <span><strong>{collection.id}</strong>{collection.description && <small>{collection.description}</small>}</span>
        <em>{collection.card_count} 卡片 · {imageCount(collection.id)} 图</em>
      </label></li>)}
    </ul> : <p className="gallery-transfer-empty">还没有收藏夹可以导出。</p>}
    <label className="gallery-transfer-option">
      <input type="checkbox" checked={download} disabled={busy} onChange={(event) => setDownload(event.target.checked)} />
      <i aria-hidden="true">{download && <Check size={12} strokeWidth={3} />}</i>
      <span><strong>同时通过浏览器下载 ZIP</strong><small>ZIP 内含 gallery-info.json，可在另一台电脑上用“导入”还原收藏夹</small></span>
    </label>
    <p className="gallery-transfer-note">确认后会在 output 目录内新建 <code>XiriusA-Gallery-日期-时间</code> 文件夹，按收藏夹分子文件夹存放图片与 gallery-info.json；画廊本身不会改变。</p>
  </>;

  const importBody = <>
    <label className="gallery-field">
      <span>ZIP 文件路径（本机）</span>
      <input value={zipPath} disabled={busy || Boolean(file)} onChange={(event) => setZipPath(event.target.value)} placeholder={file ? "已选择上传文件，路径将被忽略" : "例如 D:\\Exports\\XiriusA-Gallery-20261012-221530.zip"} spellCheck="false" />
    </label>
    {pathIssue && !file && <p className="gallery-transfer-hint">{pathIssue}</p>}
    <div className="gallery-transfer-or"><i />或<i /></div>
    <button
      type="button"
      className={`gallery-transfer-drop ${dragging ? "drop-active" : ""} ${file ? "has-file" : ""}`}
      disabled={busy}
      onClick={() => fileInput.current?.click()}
      onDragEnter={dragEnter}
      onDragOver={dragOver}
      onDragLeave={dragLeave}
      onDrop={drop}
    >
      <span>{file ? <FileArchive size={22} /> : <Upload size={22} />}</span>
      <strong>{file ? file.name : dragging ? "释放以选择这个 ZIP" : "点击选择，或把 ZIP 拖到这里"}</strong>
      <small>{file ? `${formatBytes(file.size)} · 将上传到本服务后导入` : "仅接受画廊导出的 .zip（含 gallery-info.json）"}</small>
    </button>
    {file && <button type="button" className="gallery-transfer-clear" disabled={busy} onClick={() => setFile(null)}><X size={12} />移除已选文件，改用路径</button>}
    <input ref={fileInput} type="file" accept=".zip,application/zip,application/x-zip-compressed" hidden onChange={(event) => { chooseFile(event.target.files?.[0]); event.target.value = ""; }} />
    <label className="gallery-field">
      <span>导入到 output 内的文件夹（可选）</span>
      <input value={folder} maxLength={200} disabled={busy} onChange={(event) => setFolder(event.target.value)} placeholder="留空则以当前时间命名，例如 20261012-221530" spellCheck="false" />
    </label>
    <p className="gallery-transfer-note">图片写入这个文件夹，并按 gallery-info.json 新建收藏夹与卡片；同名收藏夹已存在时会新建“名称 (2)”，不会合并或覆盖。</p>
  </>;

  const exportReceipt = receipt?.kind === "export" && <div className="gallery-transfer-receipt" role="status">
    <strong><Check size={14} />已导出 {receipt.collections.length} 个收藏夹 · {receipt.card_count} 张卡片 · {receipt.image_count} 张图片</strong>
    <p>文件夹：<code>{outputPath(receipt.output_root, receipt.folder)}</code></p>
    <p>{receipt.downloaded ? "ZIP 下载已开始；若浏览器拦截了下载，可点下方按钮重新下载。" : "未下载 ZIP；需要时可点下方按钮下载。"}</p>
    {receipt.skipped?.length > 0 && <p className="warning">{receipt.skipped.length} 张图片的精选副本已缺失，未能导出：{receipt.skipped.slice(0, 3).map((item) => `${item.collection} / ${item.image}`).join("、")}{receipt.skipped.length > 3 ? " 等" : ""}</p>}
    <a className="gallery-transfer-download" href={receipt.zip_url} download={`${receipt.folder}.zip`}><Download size={13} />下载 ZIP</a>
  </div>;

  const importReceipt = receipt?.kind === "import" && <div className="gallery-transfer-receipt" role="status">
    <strong><Check size={14} />已导入 {receipt.collections.length} 个收藏夹 · {receipt.card_count} 张卡片 · {receipt.image_count} 张图片</strong>
    <p>图片保存在：<code>{outputPath(receipt.output_root, receipt.folder)}</code></p>
    <ul>{receipt.collections.map((item) => <li key={item.id}><b>{item.id}</b>{item.renamed && <small>原名“{item.requested_id}”已存在，已另建</small>}<em>{item.card_count} 卡片 · {item.image_count} 图</em></li>)}</ul>
  </div>;

  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <div ref={dialogRef} className="gallery-small-dialog gallery-transfer-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-transfer-title" tabIndex="-1">
      <header><div><span>TRANSFER</span><h2 id="gallery-transfer-title">画廊导入 / 导出</h2></div><button type="button" aria-label="关闭导入导出" disabled={busy} onClick={onClose}><X size={18} /></button></header>
      <nav className="gallery-transfer-tabs" role="tablist" aria-label="导入或导出">
        <button type="button" role="tab" aria-selected={tab === "export"} className={tab === "export" ? "active" : ""} disabled={busy} onClick={() => switchTab("export")}><FolderOutput size={14} />导出</button>
        <button type="button" role="tab" aria-selected={tab === "import"} className={tab === "import" ? "active" : ""} disabled={busy} onClick={() => switchTab("import")}><Upload size={14} />导入</button>
      </nav>
      <div className="gallery-dialog-body gallery-transfer-body">
        {receipt ? (exportReceipt || importReceipt) : tab === "export" ? exportBody : importBody}
        {error && <p className="gallery-form-error" role="alert">{error}</p>}
      </div>
      <footer>
        {receipt ? <>
          <button type="button" onClick={() => setReceipt(null)}>{receipt.kind === "export" ? "继续导出" : "继续导入"}</button>
          <button type="button" className="primary" onClick={onClose}>完成</button>
        </> : tab === "export" ? <>
          <span>{chosen.length ? `已选 ${chosen.length} 个 · ${chosenCards} 卡片 · ${chosenImages} 图` : "未选择收藏夹"}</span>
          <button type="button" className="primary" disabled={busy || !chosen.length} onClick={() => void runExport()}>{busy ? <RefreshCw className="spin" size={14} /> : <FolderOutput size={14} />}{busy ? "正在导出" : `导出${chosen.length ? ` ${chosen.length} 个` : ""}收藏夹`}</button>
        </> : <>
          <span>{file ? "将上传所选 ZIP" : pathIssue ? "请先修正 ZIP 路径" : cleanZipPath(zipPath) ? "将读取本机路径" : "填写路径或选择 ZIP"}</span>
          <button type="button" className="primary" disabled={busy || !canImport} onClick={() => void runImport()}>{busy ? <RefreshCw className="spin" size={14} /> : <Upload size={14} />}{busy ? "正在导入" : "导入"}</button>
        </>}
      </footer>
    </div>
  </div>;
}
