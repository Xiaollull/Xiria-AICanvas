import { ArrowLeft, ChevronRight, CornerDownLeft, Folder, FolderOpen, Layers3, RefreshCw } from "lucide-react";
import { loraFileLabel } from "./lora-folders.js";

// The two pieces the generate page's LoRA modal and the standalone `/lora` page share: the engine
// bar (six engines plus a folder path field) and one folder level of a category (breadcrumbs,
// folder cards first, then the files directly inside). Rendering them once keeps the two surfaces
// from drifting — the reason `LoraMountPanel` is shared as well.

export const LORA_ENGINE_LABELS = Object.freeze({ SD: "SD", iL: "iL", Anima: "Anima", Flux: "FLUX.1", Flux2: "FLUX.2", Krea2: "Krea 2" });

export function LoraEngineBar({ engines, engine, blockedReason = "", onSelectEngine, pathValue, onPathChange, onLocate, busy = false, independent = false, placeholder = "粘贴或输入 LoRA 文件夹路径，例如 models/loras/illustrious/style" }) {
  return <div className="lora-engine-bar">
    <div className="lora-engine-switch" role="radiogroup" aria-label={independent ? "XYZ LoRA 所属引擎（只切换独立草稿）" : "LoRA 所属引擎（同时切换生图引擎）"}>
      {engines.map((name) => <button
        type="button"
        key={name}
        role="radio"
         aria-checked={engine === name}
         disabled={Boolean(blockedReason) || typeof onSelectEngine !== "function"}
         aria-disabled={Boolean(blockedReason) || typeof onSelectEngine !== "function"}
         className={engine === name ? "active" : ""}
         title={blockedReason || (engine === name ? `当前${independent ? "XYZ" : "生图"}引擎：${name}` : `切换到 ${name}（${independent ? "只切换 XYZ 独立草稿" : "文生图与图生图同步切换"}）`)}
         onClick={() => { if (!blockedReason && typeof onSelectEngine === "function") onSelectEngine(name); }}
      >{LORA_ENGINE_LABELS[name] || name}</button>)}
    </div>
    <form className="lora-path-field" onSubmit={(event) => { event.preventDefault(); onLocate(); }}>
      <FolderOpen size={14} />
      <input
        value={pathValue}
        spellCheck="false"
        aria-label="LoRA 文件夹路径"
        placeholder={placeholder}
        onChange={(event) => onPathChange(event.target.value)}
      />
      <button type="submit" disabled={busy || !String(pathValue || "").trim()} title={independent ? "定位到该文件夹（只切换 XYZ 独立草稿）" : "定位到该文件夹（属于其他引擎时会同步切换生图引擎）"}>
        {busy ? <RefreshCw className="spin" size={13} /> : <CornerDownLeft size={13} />}定位
      </button>
    </form>
  </div>;
}

export function LoraFolderGrid({ view, categoryLabel, gridClassName = "lora-library-grid", onOpenFolder, renderCard, search = "", emptyHint = "", error = "" }) {
  if (!view) return null;
  const empty = !view.folders.length && !view.files.length;
  return <div className="lora-folder-view">
    <nav className="lora-folder-crumbs" aria-label="LoRA 文件夹路径">
      <button type="button" className="lora-folder-up" disabled={view.parent === null} aria-label="返回上一级文件夹" onClick={() => onOpenFolder(view.parent ?? "")}><ArrowLeft size={13} />上一级</button>
      <button type="button" className={view.crumbs.length ? "" : "current"} onClick={() => onOpenFolder("")}>{categoryLabel}</button>
      {view.crumbs.map((crumb, index) => <span key={crumb.path}><ChevronRight size={12} /><button type="button" className={index === view.crumbs.length - 1 ? "current" : ""} aria-current={index === view.crumbs.length - 1 ? "page" : undefined} onClick={() => onOpenFolder(crumb.path)}>{crumb.name}</button></span>)}
      {view.missing && <small className="lora-folder-missing">原文件夹已不存在，显示最近的上级</small>}
    </nav>
    <div className={gridClassName}>
      {view.folders.map((folder) => <article
        key={folder.path}
        className="lora-library-card lora-folder-card"
        role="button"
        tabIndex="0"
        aria-label={`打开文件夹 ${folder.name}`}
        onClick={() => onOpenFolder(folder.path)}
        onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); onOpenFolder(folder.path); } }}
      >
        <div className="lora-folder-card-icon"><Folder size={34} /></div>
        <div className="lora-card-copy">
          <strong title={folder.name}>{folder.name}</strong>
          <small>{search ? `${folder.matches} 个匹配 · 共 ${folder.count} 个` : `${folder.count} 个 LoRA`}</small>
        </div>
      </article>)}
      {view.files.map((item) => renderCard(item, loraFileLabel(item)))}
      {empty && <div className="lora-library-empty"><Layers3 size={30} /><strong>{search ? "没有匹配的 LoRA" : "该文件夹暂无 LoRA"}</strong><p>{search ? "尝试其他关键词，或返回上一级" : emptyHint}</p></div>}
      {error && <div className="lora-library-empty error"><strong>LoRA 目录读取失败</strong><p>{error}</p></div>}
    </div>
  </div>;
}
