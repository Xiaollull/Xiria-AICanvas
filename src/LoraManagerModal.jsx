import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, ExternalLink, FolderOpen, Layers3, Maximize2, Minimize2, RefreshCw, Search, Share2, X } from "lucide-react";
import { useDialogLifecycle } from "./gallery-core.js";
import { LoraEngineBar, LoraFolderGrid } from "./LoraFolderBrowser.jsx";
import LoraMountPanel from "./LoraMountPanel.jsx";
import LoraGroupPanel from "./LoraGroupPanel.jsx";
import LoraManagerNotice from "./LoraManagerNotice.jsx";

/** One real manager DOM. Hosts own scope, catalog/navigation and all writes. */
export default function LoraManagerModal({ context: c, title = "LoRA 管理", className = "", repairEntry, children }) {
  const [localMaximized, setLocalMaximized] = useState(false);
  const maximized = c.maximized ?? localMaximized, setMaximized = c.onMaximize || setLocalMaximized;
  const ref = useDialogLifecycle(true, c.onClose);
  // Transient menus consume Escape before the parent dialog lifecycle. This also covers the
  // group creation menu: close the actual component, never a second shadow menu state.
  useEffect(() => {
    const escape = (event) => {
      if (event.key !== "Escape") return;
      const picker = ref.current?.querySelector(".workspace-select.open .workspace-select-trigger");
      const groupMenu = ref.current?.querySelector(".lora-group-create[aria-expanded=true]");
      if (!picker && !groupMenu) return;
      const pickerMenu = picker && document.getElementById(picker.getAttribute("aria-controls"));
      if (!ref.current.contains(document.activeElement) && !pickerMenu?.contains(document.activeElement)) return;
      event.preventDefault(); event.stopImmediatePropagation(); (picker || groupMenu).click();
    };
    window.addEventListener("keydown", escape, true);
    return () => window.removeEventListener("keydown", escape, true);
  }, [ref]);
  const category = c.library.find((item) => item.id === c.category);
  const matches = (item) => item.name.toLowerCase().includes(c.search.trim().toLowerCase());
  const dialog = <div className="lora-modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && c.onClose()}>
    <section className={`lora-modal ${maximized ? "maximized" : ""} ${className}`} ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title}>
      <header className="lora-modal-head">
        <div><span className="eyebrow">MODEL ASSET LIBRARY</span><h2>{title}</h2><p>{c.engine} 引擎 · 已挂载 {c.loras.length} 个</p></div>
        <div className="lora-modal-actions">
          <button type="button" className="asset-refresh" title="刷新 LoRA 列表" aria-label="刷新 LoRA 列表" disabled={c.refreshing || (c.refreshLocked ?? c.locked)} onClick={c.onRefresh}><RefreshCw className={c.refreshing ? "spin" : ""} size={16} /></button>
          <button type="button" className="modal-close" title={maximized ? "还原窗口" : "最大化窗口"} aria-label={maximized ? "还原 LoRA 管理器窗口" : "最大化 LoRA 管理器窗口"} onClick={() => setMaximized((old) => !old)}>{maximized ? <Minimize2 size={18} /> : <Maximize2 size={18} />}</button>
          <button type="button" className="modal-close" title={c.externalBlockedReason || (typeof c.onExternal === "function" ? "在新标签页打开完整 LoRA 资产页" : "当前上下文不能打开独立 LoRA 资产页")} aria-label="在新标签页打开 LoRA 管理器" disabled={Boolean(c.externalBlockedReason) || typeof c.onExternal !== "function"} aria-disabled={Boolean(c.externalBlockedReason) || typeof c.onExternal !== "function"} onClick={() => { if (!c.externalBlockedReason && typeof c.onExternal === "function") c.onExternal(); }}><ExternalLink size={18} /></button>
          <button type="button" className="modal-close" aria-label="关闭 LoRA 管理器" onClick={c.onClose}><X size={20} /></button>
        </div>
      </header>
      {repairEntry}
      <LoraEngineBar engines={c.engines} engine={c.engine} blockedReason={c.engineBlockedReason || ""} onSelectEngine={c.onEngine}
        pathValue={c.path} onPathChange={c.onPath} onLocate={c.onLocate} busy={c.pathBusy} independent={c.independent} />
      <div className="lora-modal-body">
        <nav className="lora-categories" aria-label="LoRA 分类">
          <button className={`lora-mounted-nav ${c.category === "mounted" ? "active" : ""}`} onClick={() => c.onCategory("mounted")}><span>已挂载</span><b>{c.loras.length}</b></button>
          <button className={`lora-mounted-nav ${c.category === "groups" ? "active" : ""}`} onClick={() => c.onCategory("groups")}><span>组合</span><b>{c.groups.length}</b></button>
          <span>分类</span>
          {c.library.map((item) => <button className={c.category === item.id ? "active" : ""} key={item.id} onClick={() => c.onCategory(item.id)}><span>{item.label}</span><b>{item.models.length}</b></button>)}
          {c.loading && <p>正在扫描目录...</p>}
          <div className="lora-directory" title={c.directory}><span>模型路径</span><p>{c.directory || "正在解析..."}</p></div>
        </nav>
        <div className="lora-browser">
          <div className="lora-browser-content">
            {c.category === "mounted" ? <div className="lora-mounted-manager">
              <div className="lora-browser-head"><div><h3>已挂载 LoRA 管理</h3><p>调整权重与启用状态 · 按住 Shift 可精确到 0.01 · 已启用的组合以独立分块显示</p></div></div>
              <LoraMountPanel variant="modal" loras={c.loras} groups={c.groups} locked={c.locked} cards={c.cards} onUpdateLoras={c.onLoras} onUpdateGroups={c.onGroups} onNotice={c.onNotice} preserveSnapshot={c.independent} />
            </div> : c.category === "groups" ? <div className="lora-mounted-manager">
              {c.groupLibrary}
              <LoraGroupPanel groups={c.groups} loras={c.loras} locked={c.locked} cards={c.cards} onUpdateGroups={c.onGroups} onUpdateLoras={c.onGroupLoras || c.onLoras} onNotice={c.onNotice} preserveSnapshot={c.independent}
                onToggleGroup={c.onToggleGroup} onCollectGroup={c.onCollectGroup} onCreateGroup={c.onCreateGroup} />
            </div> : <>
              <div className="lora-browser-head"><div><h3>{category?.label || "LoRA"}</h3><p>{category?.directory ? `${category.directory}${c.folder?.path ? `/${c.folder.path}` : ""}` : c.error || "选择分类浏览模型"}</p></div><label className="lora-search"><Search size={15} /><input value={c.search} onChange={(event) => c.onSearch(event.target.value)} placeholder={c.folder?.path ? "在当前文件夹中搜索" : "搜索 LoRA"} /></label></div>
              {category?.shared ? <div className="lora-modal-shared-view">
                {(category.roots || []).map((root) => {
                  const folders = (root.folders || []).map((folder) => ({ ...folder, models: folder.models.filter(matches) })).filter((folder) => folder.models.length);
                  return <section className="lora-modal-shared-root" key={root.id}>
                    <header><div><Share2 size={15} /><strong>{root.label}</strong><code title={root.path}>{root.path}</code></div><span>{root.files} 个文件 · {c.formatBytes(root.bytes)}</span></header>
                    {folders.length ? folders.map((folder) => {
                      const key = `${root.id}/${folder.name}`, collapsed = c.collapsedFolders[key] !== false;
                      return <div className={`lora-modal-shared-folder ${collapsed ? "collapsed" : ""}`} key={key}>
                        <button type="button" aria-expanded={!collapsed} onClick={() => c.onCollapsedFolders((old) => ({ ...old, [key]: !collapsed }))}><ChevronDown size={14} /><FolderOpen size={15} /><strong>{folder.label}</strong><b>{folder.models.length}</b></button>
                        {!collapsed && <div className="lora-library-grid">{folder.models.map((item) => c.renderCard(item, folder.label))}</div>}
                      </div>;
                    }) : <p className="lora-modal-shared-empty">{c.search ? "该共享目录中没有匹配的文件" : "该共享目录为空"}</p>}
                  </section>;
                })}
              </div> : c.folder ? <LoraFolderGrid view={c.folder} categoryLabel={category.label} search={c.search} onOpenFolder={c.onFolder}
                renderCard={(item, fileLabel) => c.renderCard(item, c.folder.path ? `${category.label}/${c.folder.path}` : category.label, fileLabel)} emptyHint={`将模型文件放入 ${category?.directory || "对应分类目录"}`} error={c.error} /> : <div className="lora-library-grid">
                {(category?.models || []).filter(matches).map((item) => c.renderCard(item, category?.label || ""))}
                {!category?.models.filter(matches).length && <div className="lora-library-empty"><Layers3 size={30} /><strong>{c.search ? "没有匹配的 LoRA" : "该分类暂无 LoRA"}</strong><p>{c.search ? "尝试其他关键词" : `将模型文件放入 ${category?.directory || "对应分类目录"}`}</p></div>}
                {c.error && <div className="lora-library-empty error"><X size={30} /><strong>LoRA 目录读取失败</strong><p>{c.error}</p></div>}
              </div>}
            </>}
          </div>
          <LoraManagerNotice notice={c.notice} onDismiss={c.onDismissNotice} />
        </div>
      </div>
      <footer className="lora-modal-foot"><span>{c.footer || "在已挂载标签管理权重和状态；在分类标签浏览并挂载或卸载 LoRA"}</span><div className="lora-modal-foot-actions">{c.onConfirm && <button type="button" className="secondary" onClick={c.onClose}>取消</button>}<button type="button" className="primary" onClick={c.onConfirm || c.onClose}>{c.confirmLabel || `完成 · ${c.loras.length} 已挂载`}</button></div></footer>
      {children}
    </section>
  </div>;
  return c.independent ? createPortal(dialog, document.body) : dialog;
}
