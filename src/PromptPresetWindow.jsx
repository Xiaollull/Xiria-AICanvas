// The prompt preset library as a floating window, shared by 文生图 and 图生图.
//
// Two modules — positive and negative — each hold the groups the user names, and each
// group holds prompt cards. A card switched on is composed onto the request when a job is submitted;
// it is never written into the prompt box (`composeGenerationPrompt`). Groups and cards are created,
// renamed, deleted and dragged into order here; a card's own text and preview picture are edited in
// `PromptPresetEditor`, text on the left and picture on the right. The window floats over the page
// like the task queue — portalled to the document, dragged by its header, kept on screen — so it can
// stay open beside the prompt it feeds.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  GripVertical,
  ImageIcon,
  ImagePlus,
  MoreHorizontal,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import { useDialogLifecycle } from "./gallery-core";
import { useFloatingWindowLayer } from "./floating-windows.js";
import { HOVER_PREVIEW_DELAY, hoverPreviewPlacement } from "./lora-hover-preview.js";
import { prepareCardImage, readImageFileAsDataUrl } from "./lora-card-image.js";
import { clampQueueWindow, queueWindowDrag } from "./queue-window.js";
import {
  PROMPT_PRESET_GROUP_NAME_LIMIT,
  PROMPT_PRESET_NAME_LIMIT,
  PROMPT_PRESET_TYPES,
  promptPresetGroupCards,
  promptPresetPreviewUrl,
  sortPromptPresetGroups,
  sortPromptPresetRecords,
  validatePromptPresetDraft,
} from "./prompt-presets.js";

export const PRESET_WINDOW_SIZE = Object.freeze({ width: 560, height: 510 });
const PREVIEW_MAX_EDGE = 768;
const PREVIEW_SOURCE_BYTES = 20 * 1024 * 1024;
const SORT_THRESHOLD = 5;
const STRIP_CHIP_LIMIT = 6;

export const PRESET_MODULE_LABELS = Object.freeze({ positive: "正向", negative: "负向" });
const POSITION_LABELS = Object.freeze({ start: "开头", middle: "中间", end: "结尾" });

// ---------------------------------------------------------------------------------------------
// Hover thumbnail: the card's picture, shown beside it while the pointer rests on it.

export function usePresetPreviewHover({ enabled = true } = {}) {
  const [preview, setPreview] = useState(null);
  const timerRef = useRef(null);
  const hide = useCallback(() => {
    window.clearTimeout(timerRef.current);
    setPreview((current) => (current === null ? current : null));
  }, []);
  const show = useCallback((element, record) => {
    window.clearTimeout(timerRef.current);
    const url = promptPresetPreviewUrl(record?.preview);
    if (!enabled || !element || !url) return;
    const rect = element.getBoundingClientRect();
    const anchor = { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
    timerRef.current = window.setTimeout(() => setPreview({ anchor, url, name: record.name, width: record.preview.width, height: record.preview.height }), HOVER_PREVIEW_DELAY);
  }, [enabled]);
  useEffect(() => () => window.clearTimeout(timerRef.current), []);
  useEffect(() => { if (!enabled) hide(); }, [enabled, hide]);
  const bind = useCallback((record) => ({
    onPointerEnter: (event) => show(event.currentTarget, record),
    onPointerLeave: hide,
    onFocus: (event) => show(event.currentTarget, record),
    onBlur: hide,
  }), [hide, show]);
  return { preview, hide, bind };
}

export function PresetPreviewHover({ preview }) {
  const panelRef = useRef(null);
  const [placement, setPlacement] = useState(null);
  const place = useCallback(() => {
    const panel = panelRef.current;
    if (!panel || !preview) return;
    const rect = panel.getBoundingClientRect();
    setPlacement(hoverPreviewPlacement({ anchor: preview.anchor, viewport: { width: window.innerWidth, height: window.innerHeight }, size: { width: rect.width, height: rect.height } }));
  }, [preview]);
  useLayoutEffect(() => {
    if (!preview) {
      setPlacement(null);
      return undefined;
    }
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [preview, place]);
  if (!preview) return null;
  const style = placement ? { left: `${placement.left}px`, top: `${placement.top}px` } : { left: "0px", top: "0px", visibility: "hidden" };
  // The recorded size reserves the box before the picture loads, so it is placed once, correctly.
  const ratio = preview.width && preview.height ? `${preview.width} / ${preview.height}` : undefined;
  return createPortal(
    <div ref={panelRef} className="preset-preview-hover" style={style} aria-hidden="true">
      <img src={preview.url} alt="" style={ratio ? { aspectRatio: ratio } : undefined} onLoad={place} />
      <span>{preview.name}</span>
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------------------------
// The strip under each prompt box: what is switched on, and the way into the window.

export function PromptPresetStrip({ type, container, activeIds, open, disabled = false, libraryError = "", onOpen, onToggle }) {
  const hover = usePresetPreviewHover();
  const cards = sortPromptPresetRecords(container?.records, type);
  const active = cards.filter((record) => activeIds.includes(record.id));
  const shown = active.slice(0, STRIP_CHIP_LIMIT);
  return <div className={`preset-strip ${type} ${open ? "open" : ""}`}>
    <span className="preset-strip-title">{PRESET_MODULE_LABELS[type]}预设</span>
    <div className="preset-strip-chips">
      {libraryError ? <small className="preset-strip-error">预设库不可用</small>
        : active.length === 0 ? <small>未选择</small>
          : shown.map((record) => <span className="preset-strip-chip" key={record.id} {...hover.bind(record)}>
            {record.preview && <ImageIcon size={11} aria-hidden="true" />}
            <b>{record.name}</b>
            <button type="button" aria-label={`停用预设“${record.name}”`} title="停用" disabled={disabled} onClick={() => onToggle(record)}><X size={11} /></button>
          </span>)}
      {active.length > shown.length && <small>+{active.length - shown.length}</small>}
    </div>
    <button type="button" className="preset-strip-open" data-prompt-preset-focus-fallback={type} aria-label={`${open ? "收起" : "管理"}${PRESET_MODULE_LABELS[type]}预设，已选 ${active.length} 项，共 ${cards.length} 项`} aria-expanded={open} onClick={(event) => onOpen(type, event.currentTarget)}>
      {open ? "收起" : "管理预设"}{active.length > 0 && <b>{active.length}</b>}<ChevronDown size={13} aria-hidden="true" />
    </button>
    <PresetPreviewHover preview={hover.preview} />
  </div>;
}

// ---------------------------------------------------------------------------------------------
// Pointer sorting of groups (chips) and cards, with the element under the pointer as the target.

function sortTarget(drag, x, y) {
  const element = document.elementFromPoint(x, y);
  if (!element) return null;
  const card = element.closest("[data-preset-card-id]");
  const chip = element.closest("[data-preset-group-id]");
  if (drag.kind === "card") {
    if (card && card.dataset.presetCardId !== drag.id) {
      const rect = card.getBoundingClientRect();
      return { kind: "card", id: card.dataset.presetCardId, groupId: card.dataset.presetCardGroup, side: y < rect.top + rect.height / 2 ? "before" : "after" };
    }
    if (chip) return { kind: "group", id: chip.dataset.presetGroupId };
    return null;
  }
  if (chip && chip.dataset.presetGroupId !== drag.id) {
    const rect = chip.getBoundingClientRect();
    return { kind: "group", id: chip.dataset.presetGroupId, side: x < rect.left + rect.width / 2 ? "before" : "after" };
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// The window.

export function PromptPresetWindow({
  container,
  activeIds,
  module,
  selectedGroups,
  position,
  disabled = false,
  libraryError = "",
  libraryWarning = "",
  onModule,
  onSelectGroup,
  onMove,
  onClose,
  onToggle,
  onCreateGroup,
  onRenameGroup,
  onDeleteGroup,
  onReorderGroup,
  onCreateCard,
  onEditCard,
  onDeleteCard,
  onReorderCard,
}) {
  const panelRef = useRef(null);
  const dragRef = useRef(null);
  const sortRef = useRef(null);
  const suppressClickRef = useRef(false);
  const menuRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [sort, setSort] = useState(null);
  const [menuId, setMenuId] = useState("");
  const [newGroup, setNewGroup] = useState(null);
  const [renaming, setRenaming] = useState(null);
  const [message, setMessage] = useState("");
  const locked = disabled || Boolean(libraryError);
  const hover = usePresetPreviewHover({ enabled: !sort });
  // Shares a stack with the task queue: whichever was opened, clicked or dragged last is on top.
  const layer = useFloatingWindowLayer("presets");

  const groups = sortPromptPresetGroups(container, module);
  const selected = groups.find((group) => group.id === selectedGroups?.[module]) || groups[0] || null;
  const cards = selected ? promptPresetGroupCards(container, selected.id) : [];
  const counts = Object.fromEntries(PROMPT_PRESET_TYPES.map((type) => {
    const all = sortPromptPresetRecords(container?.records, type);
    return [type, { total: all.length, active: all.filter((record) => activeIds[type]?.includes(record.id)).length }];
  }));

  const viewport = () => ({ width: window.innerWidth, height: window.innerHeight });
  const panelSize = () => {
    const panel = panelRef.current;
    return panel ? { width: panel.offsetWidth, height: panel.offsetHeight } : PRESET_WINDOW_SIZE;
  };
  useLayoutEffect(() => {
    const settled = clampQueueWindow(position, panelSize(), viewport());
    if (settled.x !== position?.x || settled.y !== position?.y) onMove(settled);
  }, [position, onMove]);
  useEffect(() => {
    const onResize = () => onMove(clampQueueWindow(position, panelSize(), viewport()));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [position, onMove]);
  useEffect(() => {
    if (!menuId) return undefined;
    const close = (event) => { if (!menuRef.current?.contains(event.target)) setMenuId(""); };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [menuId]);
  // A new module or group starts with nothing half-edited from the last one.
  useEffect(() => { setRenaming(null); setMenuId(""); setMessage(""); }, [module, selected?.id]);

  const startWindowDrag = (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    dragRef.current = { pointerId: event.pointerId, x: position.x, y: position.y, pointerX: event.clientX, pointerY: event.clientY };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setDragging(true);
  };
  const moveWindowDrag = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    onMove(queueWindowDrag(drag, { x: event.clientX, y: event.clientY }, panelSize(), viewport()));
  };
  const endWindowDrag = (event) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  };

  const beginSort = (event, kind, item) => {
    if (locked || event.button !== 0) return;
    sortRef.current = { kind, id: item.id, label: item.name, pointerId: event.pointerId, x0: event.clientX, y0: event.clientY, active: false, target: null };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const moveSort = (event) => {
    const drag = sortRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.active && Math.hypot(event.clientX - drag.x0, event.clientY - drag.y0) < SORT_THRESHOLD) return;
    event.preventDefault();
    drag.active = true;
    drag.target = sortTarget(drag, event.clientX, event.clientY);
    setSort({ kind: drag.kind, id: drag.id, label: drag.label, x: event.clientX, y: event.clientY, target: drag.target });
  };
  const cancelSort = () => {
    sortRef.current = null;
    setSort(null);
  };
  const endSort = (event) => {
    const drag = sortRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    cancelSort();
    if (!drag.active) return;
    // The click that follows the release belongs to the drag, not to selecting a group.
    suppressClickRef.current = true;
    window.setTimeout(() => { suppressClickRef.current = false; }, 0);
    const target = drag.target;
    if (!target) return;
    const run = (action) => {
      const error = action();
      setMessage(typeof error === "string" ? error : "");
    };
    if (drag.kind === "card") {
      if (target.kind === "card") {
        const siblings = promptPresetGroupCards(container, target.groupId).filter((record) => record.id !== drag.id);
        const index = siblings.findIndex((record) => record.id === target.id);
        run(() => onReorderCard(drag.id, target.side === "before" ? index : index + 1, target.groupId));
      } else {
        run(() => onReorderCard(drag.id, promptPresetGroupCards(container, target.id).filter((record) => record.id !== drag.id).length, target.id));
      }
      return;
    }
    const siblings = groups.filter((group) => group.id !== drag.id);
    const index = siblings.findIndex((group) => group.id === target.id);
    run(() => onReorderGroup(drag.id, target.side === "before" ? index : index + 1));
  };
  const sortHandlers = (kind, item) => ({
    onPointerDown: (event) => beginSort(event, kind, item),
    onPointerMove: moveSort,
    onPointerUp: endSort,
    onPointerCancel: () => cancelSort(),
  });
  const dropClass = (kind, id) => (sort?.target?.kind === kind && sort.target.id === id ? `drop-${sort.target.side || "into"}` : "");

  const submitNewGroup = (event) => {
    event.preventDefault();
    const result = onCreateGroup(module, newGroup?.name || "");
    if (result?.error) {
      setNewGroup((current) => ({ ...current, error: result.error }));
      return;
    }
    setNewGroup(null);
    if (result?.group) onSelectGroup(module, result.group.id);
  };
  const submitRename = (event) => {
    event.preventDefault();
    const error = onRenameGroup(renaming.id, renaming.name);
    if (error) setRenaming((current) => ({ ...current, error }));
    else setRenaming(null);
  };
  const moveGroup = (step) => {
    const index = groups.findIndex((group) => group.id === selected.id);
    setMessage(onReorderGroup(selected.id, Math.max(0, index + step)) || "");
  };
  const moveCard = (record, step) => {
    const index = cards.findIndex((card) => card.id === record.id);
    setMessage(onReorderCard(record.id, Math.max(0, index + step), record.groupId) || "");
  };
  const keyDown = (event) => {
    if (event.key !== "Escape") return;
    event.stopPropagation();
    if (sortRef.current) cancelSort();
    else if (menuId) setMenuId("");
    else if (renaming) setRenaming(null);
    else if (newGroup) setNewGroup(null);
    else onClose();
  };

  return createPortal(
    <div
      ref={panelRef}
      className={`preset-window ${module} ${dragging ? "dragging" : ""} ${sort ? "sorting" : ""}`}
      role="dialog"
      aria-label="提示词预设"
      style={{ left: `${position?.x ?? 0}px`, top: `${position?.y ?? 0}px`, zIndex: layer.zIndex }}
      onKeyDown={keyDown}
      {...layer.bind}
    >
      <header onPointerDown={startWindowDrag} onPointerMove={moveWindowDrag} onPointerUp={endWindowDrag} onPointerCancel={endWindowDrag}>
        <div><GripVertical size={15} aria-hidden="true" /><strong>提示词预设</strong><small>已选 {counts.positive.active + counts.negative.active} 项</small></div>
        <button type="button" className="icon-button" title="关闭预设窗口" aria-label="关闭预设窗口" onClick={onClose}><X size={15} /></button>
      </header>
      <nav className="preset-window-modules" role="tablist" aria-label="预设模块">
        {PROMPT_PRESET_TYPES.map((type, index) => <button type="button" role="tab" key={type} id={`preset-tab-${type}`} aria-controls="preset-library-panel" tabIndex={module === type ? 0 : -1} aria-selected={module === type} className={`${type} ${module === type ? "active" : ""}`} onClick={() => onModule(type)} onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? PROMPT_PRESET_TYPES[0] : event.key === "End" ? PROMPT_PRESET_TYPES.at(-1) : PROMPT_PRESET_TYPES[(index + 1) % PROMPT_PRESET_TYPES.length];
          onModule(next);
          document.getElementById(`preset-tab-${next}`)?.focus();
        }}>
          <strong>{PRESET_MODULE_LABELS[type]}</strong><small>{counts[type].active > 0 ? `${counts[type].active} / ${counts[type].total}` : counts[type].total}</small>
        </button>)}
      </nav>
      <div className="preset-library-panel" id="preset-library-panel" role="tabpanel" aria-labelledby={`preset-tab-${module}`}>
      {libraryError && <p className="preset-window-note error" role="alert">{libraryError}</p>}
      {!libraryError && libraryWarning && <p className="preset-window-note" role="status">{libraryWarning}</p>}
      <div className="preset-window-groups" role="toolbar" aria-label={`${PRESET_MODULE_LABELS[module]}分组，可拖动排序`}>
        {groups.map((group) => {
          const groupCards = promptPresetGroupCards(container, group.id);
          const on = groupCards.filter((record) => activeIds[module]?.includes(record.id)).length;
          return <button
            type="button"
            key={group.id}
            data-preset-group-id={group.id}
            className={`preset-group-chip ${group.id === selected?.id ? "selected" : ""} ${sort?.id === group.id ? "dragging" : ""} ${dropClass("group", group.id)}`}
            aria-pressed={group.id === selected?.id}
            title={`${group.name}：${groupCards.length} 张卡片${on ? `，${on} 张已启用` : ""}（拖动可排序）`}
            onClick={() => { if (!suppressClickRef.current) onSelectGroup(module, group.id); }}
            {...sortHandlers("group", group)}
          >
            <span>{group.name}</span><b>{on ? `${on}/` : ""}{groupCards.length}</b>
          </button>;
        })}
        {newGroup
          ? <form className="preset-group-new" onSubmit={submitNewGroup}>
            <input autoFocus value={newGroup.name} maxLength={PROMPT_PRESET_GROUP_NAME_LIMIT} placeholder="分组名称，如 角色" aria-label="新分组名称" aria-invalid={Boolean(newGroup.error)} onChange={(event) => setNewGroup({ name: event.target.value, error: "" })} />
            <button type="submit" aria-label="创建分组" title="创建"><Check size={13} /></button>
            <button type="button" aria-label="取消新建分组" title="取消" onClick={() => setNewGroup(null)}><X size={13} /></button>
          </form>
          : <button type="button" className="preset-group-add" disabled={locked} onClick={() => setNewGroup({ name: "", error: "" })}><Plus size={13} />分组</button>}
      </div>
      {newGroup?.error && <p className="preset-window-note error" role="alert">{newGroup.error}</p>}
      {selected && <div className="preset-group-bar">
        {renaming
          ? <form className="preset-group-rename" onSubmit={submitRename}>
            <input autoFocus value={renaming.name} maxLength={PROMPT_PRESET_GROUP_NAME_LIMIT} aria-label="分组名称" aria-invalid={Boolean(renaming.error)} onChange={(event) => setRenaming({ ...renaming, name: event.target.value, error: "" })} />
            <button type="submit" title="保存名称" aria-label="保存分组名称"><Check size={13} /></button>
            <button type="button" title="取消" aria-label="取消重命名" onClick={() => setRenaming(null)}><X size={13} /></button>
          </form>
          : <div className="preset-group-name"><small>{cards.length} 项预设</small></div>}
        <div className="preset-group-actions">
          <button type="button" title="重命名分组" aria-label={`重命名分组“${selected.name}”`} disabled={locked || Boolean(renaming)} onClick={() => setRenaming({ id: selected.id, name: selected.name, error: "" })}><Pencil size={13} /></button>
          <button type="button" title="分组左移" aria-label="分组左移" disabled={locked || groups[0]?.id === selected.id} onClick={() => moveGroup(-1)}><ArrowLeft size={13} /></button>
          <button type="button" title="分组右移" aria-label="分组右移" disabled={locked || groups.at(-1)?.id === selected.id} onClick={() => moveGroup(1)}><ArrowRight size={13} /></button>
          <button type="button" className="danger" title="删除分组" aria-label={`删除分组“${selected.name}”`} disabled={locked} onClick={() => onDeleteGroup(selected)}><Trash2 size={13} /></button>
          <button type="button" className="primary" disabled={locked} onClick={() => onCreateCard(module, selected.id)}><Plus size={13} />新建预设</button>
        </div>
      </div>}
      {renaming?.error && <p className="preset-window-note error" role="alert">{renaming.error}</p>}
      {message && <p className="preset-window-note error" role="alert">{message}</p>}
      <div className="preset-window-body">
        {!groups.length && <div className="preset-window-empty">
          <strong>还没有{PRESET_MODULE_LABELS[module]}预设</strong>
          <p>新建分组，整理常用的角色、画风或质量提示词。</p>
          <button type="button" disabled={locked} onClick={() => setNewGroup({ name: "", error: "" })}><Plus size={13} />新建分组</button>
        </div>}
        {selected && !cards.length && <div className="preset-window-empty">
          <strong>“{selected.name}”里还没有预设</strong>
          <p>保存一段常用提示词，下次生成时勾选使用。</p>
          <button type="button" disabled={locked} onClick={() => onCreateCard(module, selected.id)}><Plus size={13} />新建预设</button>
        </div>}
        {cards.length > 0 && <div className="preset-card-grid">
          {cards.map((record, index) => {
            const on = activeIds[module]?.includes(record.id);
            const menuOpen = menuId === record.id;
            return <article
              key={record.id}
              data-preset-card-id={record.id}
              data-preset-card-group={record.groupId}
              className={`preset-card ${on ? "active" : ""} ${sort?.id === record.id ? "dragging" : ""} ${dropClass("card", record.id)}`}
              {...hover.bind(record)}
            >
              <button type="button" className="preset-card-grip" aria-label={`拖动排序：${record.name}`} title="拖动排序，可拖到其他分组" disabled={locked} {...sortHandlers("card", record)}><GripVertical size={14} /></button>
              <button type="button" className="preset-card-main" aria-pressed={on} disabled={locked} aria-label={`${on ? "停用" : "启用"}${PRESET_MODULE_LABELS[module]}预设“${record.name}”（生成时置于${POSITION_LABELS[record.position]}）`} onClick={() => onToggle(record)}>
                <span><i className="preset-card-check" aria-hidden="true">{on && <Check size={12} />}</i><strong>{record.name}</strong>{record.preview && <ImageIcon size={12} aria-label="有预览图" />}<b>{POSITION_LABELS[record.position]}</b></span>
                <small>{record.content.replace(/\s+/g, " ")}</small>
              </button>
              <button type="button" className="preset-card-edit" aria-label={`编辑预设：${record.name}`} title="编辑预设" disabled={locked} onClick={() => { hover.hide(); onEditCard(record); }}><Pencil size={14} /></button>
              <div className="preset-card-more-wrap" ref={menuOpen ? menuRef : undefined}>
                <button type="button" className="preset-card-more" aria-label={`更多操作：${record.name}`} aria-haspopup="menu" aria-expanded={menuOpen} disabled={locked} onClick={() => setMenuId((current) => (current === record.id ? "" : record.id))}><MoreHorizontal size={16} aria-hidden="true" /></button>
                {menuOpen && <div className="preset-card-menu" role="menu" aria-label={`${record.name}卡片操作`}>
                  <button type="button" role="menuitem" onClick={() => { setMenuId(""); onEditCard(record); }}><Pencil size={13} />编辑</button>
                  <button type="button" role="menuitem" disabled={index === 0} onClick={() => { setMenuId(""); moveCard(record, -1); }}><ArrowLeft size={13} />前移</button>
                  <button type="button" role="menuitem" disabled={index === cards.length - 1} onClick={() => { setMenuId(""); moveCard(record, 1); }}><ArrowRight size={13} />后移</button>
                  <button type="button" role="menuitem" className="danger" onClick={() => { setMenuId(""); onDeleteCard(record); }}><Trash2 size={13} />删除</button>
                </div>}
              </div>
            </article>;
          })}
        </div>}
      </div>
      </div>
      <footer><small>勾选后在生成时拼接，不改动输入框。</small><small title="拖动左侧手柄调整顺序，拖到分组上可移入该分组；也可通过更多菜单前移、后移。">拖动手柄排序</small></footer>
      {/* Drawn into the document, not the window: the window is its own stacking layer, and the
          ghost must stay above the other floating window when it is dragged across it. */}
      {sort && createPortal(<div className="preset-sort-ghost" style={{ left: `${sort.x + 14}px`, top: `${sort.y + 10}px` }} aria-hidden="true">{sort.label}</div>, document.body)}
      <PresetPreviewHover preview={hover.preview} />
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------------------------
// The card editor: text on the left, preview picture on the right.

async function uploadPresetPreview(file) {
  if (!/^image\/(?:png|jpeg|webp|gif)$/.test(file?.type || "")) throw new Error("预览图只支持 PNG、JPEG、WebP 或 GIF");
  if (file.size > PREVIEW_SOURCE_BYTES) throw new Error("预览图不能超过 20 MiB");
  // Downscaled in the page: a card is never drawn large, and the original never crosses the wire.
  const prepared = await prepareCardImage(await readImageFileAsDataUrl(file), { maxEdge: PREVIEW_MAX_EDGE });
  const response = await fetch("/api/prompt-presets/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataUrl: prepared.dataUrl }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.id) throw new Error(payload.error || "预览图保存失败");
  return { id: payload.id, width: prepared.width, height: prepared.height };
}

export function PromptPresetEditor({ dialog, container, disabled = false, onSave, onRequestClose }) {
  const record = dialog.record || null;
  const original = record
    ? { name: record.name, content: record.content, position: record.position, groupId: record.groupId, preview: record.preview || null }
    : { name: "", content: "", position: "end", groupId: dialog.groupId || "", preview: null };
  const [draft, setDraft] = useState(original);
  const [operationError, setOperationError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef(null);
  const type = container.groups.find((group) => group.id === draft.groupId)?.type || dialog.type || "positive";
  const groups = sortPromptPresetGroups(container, type);
  const validation = validatePromptPresetDraft(draft, container, record?.id || null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(original);
  const requestClose = () => {
    if (uploading) return;
    if (dirty && !window.confirm("卡片尚未保存，确定要放弃更改吗？")) return;
    onRequestClose();
  };
  const dialogRef = useDialogLifecycle(true, requestClose, `[data-prompt-preset-focus-fallback="${type}"]`);
  const update = (key, value) => setDraft((current) => ({ ...current, [key]: value }));
  const attach = async (file) => {
    if (!file || disabled || uploading) return;
    setUploading(true);
    setUploadError("");
    try {
      update("preview", await uploadPresetPreview(file));
    } catch (error) {
      setUploadError(error.message || "预览图保存失败");
    } finally {
      setUploading(false);
    }
  };
  const previewUrl = promptPresetPreviewUrl(draft.preview);
  const titleId = `prompt-preset-editor-title-${record ? "edit" : "new"}`;
  return <div className="prompt-preset-backdrop" onMouseDown={(event) => event.target === event.currentTarget && requestClose()}>
    <form
      ref={dialogRef}
      className={`prompt-preset-editor ${type}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      tabIndex="-1"
      onSubmit={(event) => {
        event.preventDefault();
        if (validation.valid && !uploading && !disabled) setOperationError(onSave(draft) || "");
      }}
    >
      <header>
        <div><h2 id={titleId}>{record ? "编辑预设" : "新建预设"}</h2><span className="prompt-preset-editor-type">{PRESET_MODULE_LABELS[type]}提示词</span></div>
        <button type="button" aria-label="关闭卡片编辑" onClick={requestClose}><X size={18} /></button>
      </header>
      <div className="prompt-preset-editor-body">
        <fieldset className="prompt-preset-editor-fields" disabled={disabled}>
          {operationError && <p className="prompt-preset-error" role="alert">{operationError}</p>}
          <label className={validation.errors.name && draft.name ? "invalid" : ""}><span>名称 <small>{draft.name.length} / {PROMPT_PRESET_NAME_LIMIT}</small></span><input autoFocus data-dialog-autofocus value={draft.name} maxLength={PROMPT_PRESET_NAME_LIMIT} aria-invalid={Boolean(validation.errors.name)} onChange={(event) => update("name", event.target.value)} placeholder="例如：电影感人像" />{validation.errors.name && draft.name && <em>{validation.errors.name}</em>}</label>
          <div className="prompt-preset-editor-row">
            <label><span>分组</span><select value={draft.groupId} aria-invalid={Boolean(validation.errors.groupId)} onChange={(event) => update("groupId", event.target.value)}>{!groups.some((group) => group.id === draft.groupId) && <option value="">请选择分组</option>}{groups.map((group) => <option value={group.id} key={group.id}>{group.name}</option>)}</select></label>
            <fieldset><legend>拼接位置</legend><div className="prompt-preset-segmented three">{Object.entries(POSITION_LABELS).map(([value, label]) => <button type="button" className={draft.position === value ? "active" : ""} aria-pressed={draft.position === value} key={value} onClick={() => update("position", value)}>{label}</button>)}</div></fieldset>
          </div>
          <label className="prompt-preset-editor-content"><span>提示词内容 <small>{draft.content.length} 字符</small></span><textarea value={draft.content} aria-invalid={Boolean(validation.errors.content)} onChange={(event) => update("content", event.target.value)} placeholder="输入常用提示词，支持权重、括号与换行" /></label>
          <small className="prompt-preset-editor-hint">生成时按「开头 → 中间 → 输入框 → 结尾」拼接，各段空一行，不改动输入框。</small>
        </fieldset>
        <section className="prompt-preset-editor-preview" aria-label="预览图">
          <div className="prompt-preset-preview-label">预览图 <small>选填</small></div>
          <div
            className={`prompt-preset-preview-box ${previewUrl ? "filled" : ""} ${dragOver ? "drop-active" : ""}`}
            role="button"
            tabIndex={disabled ? -1 : 0}
            aria-label={previewUrl ? "更换预览图：点击选择图片，或拖入、粘贴图片" : "添加预览图：点击选择图片，或拖入、粘贴图片"}
            aria-busy={uploading}
            onClick={() => !uploading && !disabled && fileInput.current?.click()}
            onKeyDown={(event) => { if ((event.key === "Enter" || event.key === " ") && !uploading && !disabled) { event.preventDefault(); fileInput.current?.click(); } }}
            onDragOver={(event) => { if ([...(event.dataTransfer?.types || [])].includes("Files")) { event.preventDefault(); setDragOver(true); } }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => { if (![...(event.dataTransfer?.types || [])].includes("Files")) return; event.preventDefault(); setDragOver(false); void attach(event.dataTransfer.files?.[0]); }}
            onPaste={(event) => { const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/")); if (file) { event.preventDefault(); void attach(file); } }}
          >
            {previewUrl
              ? <img src={previewUrl} alt={`${draft.name || "卡片"}的预览图`} />
              : <span className="prompt-preset-preview-empty"><ImagePlus size={26} /><strong>添加预览图</strong><small>点击选择、拖入或粘贴图片</small></span>}
            {uploading && <span className="prompt-preset-preview-busy"><RefreshCw className="spin" size={20} />正在处理图片</span>}
          </div>
          <input ref={fileInput} type="file" hidden accept="image/png,image/jpeg,image/webp,image/gif" onChange={(event) => { void attach(event.target.files?.[0]); event.target.value = ""; }} />
          <div className="prompt-preset-preview-actions">
            <button type="button" disabled={uploading || disabled} onClick={() => fileInput.current?.click()}><Upload size={13} />{previewUrl ? "更换图片" : "选择图片"}</button>
            {previewUrl && <button type="button" className="danger" disabled={uploading || disabled} onClick={() => update("preview", null)}><Trash2 size={13} />移除</button>}
          </div>
          {uploadError && <p className="prompt-preset-error" role="alert">{uploadError}</p>}
          <small>悬停预设时显示。支持点击、拖入或粘贴；保存时缩小至 {PREVIEW_MAX_EDGE} px 以内。</small>
        </section>
      </div>
      <footer><span>{dirty ? "有未保存的更改" : ""}</span><div><button type="button" onClick={requestClose}>取消</button><button type="submit" className="primary" disabled={!validation.valid || uploading || disabled}><Save size={14} />保存预设</button></div></footer>
    </form>
  </div>;
}

// ---------------------------------------------------------------------------------------------
// Confirmation for deleting a card or a whole group.

export function PromptPresetConfirm({ title, detail, confirmLabel = "确认删除", focusFallback = "", onConfirm, onClose }) {
  const dialogRef = useDialogLifecycle(true, onClose, focusFallback);
  return <div className="prompt-preset-backdrop prompt-preset-delete-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="prompt-preset-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="prompt-preset-confirm-title" tabIndex="-1">
      <Trash2 size={25} /><h2 id="prompt-preset-confirm-title">{title}</h2><p>{detail}</p>
      <div><button type="button" autoFocus onClick={onClose}>取消</button><button type="button" className="danger" onClick={onConfirm}>{confirmLabel}</button></div>
    </section>
  </div>;
}
